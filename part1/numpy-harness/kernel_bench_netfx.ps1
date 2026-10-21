# JIT-compiled (RyuJIT x64, .NET Framework 4.8) SCALAR throughput of the segment–capsule kernel used by
# dense_melee_bench.py, plus the bounding-sphere prefilter. Runs inside Windows PowerShell 5.1 via Add-Type.
# Wide SIMD (Vector<T>) is not in the in-box .NET Framework assemblies, so SIMD speed-up is taken from literature.
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\research\dense_melee_bench\kernel_bench_netfx.ps1

$code = @'
using System;
using System.Diagnostics;
using System.Text;

public static class KernelBench
{
    static float[] px, py, pz, dx, dy, dz, qx, qy, qz, ex, ey, ez, rc, rb;
    static byte[] hitS;
    static int n;

    static void Fill(int count)
    {
        n = count;
        var r = new Random(20260916);
        px = new float[n]; py = new float[n]; pz = new float[n]; dx = new float[n]; dy = new float[n]; dz = new float[n];
        qx = new float[n]; qy = new float[n]; qz = new float[n]; ex = new float[n]; ey = new float[n]; ez = new float[n];
        rc = new float[n]; rb = new float[n]; hitS = new byte[n];
        for (int i = 0; i < n; i++)
        {
            px[i] = (float)(r.NextDouble() * 2); py[i] = (float)(r.NextDouble() * 2); pz[i] = (float)(r.NextDouble() * 2);
            dx[i] = (float)(r.NextDouble() * 2 - 1); dy[i] = (float)(r.NextDouble() * 2 - 1); dz[i] = (float)(r.NextDouble() * 2 - 1);
            qx[i] = (float)(r.NextDouble() * 2); qy[i] = (float)(r.NextDouble() * 2); qz[i] = (float)(r.NextDouble() * 2);
            ex[i] = (float)(r.NextDouble() * 0.8 - 0.4); ey[i] = (float)(r.NextDouble() * 0.8 - 0.4); ez[i] = (float)(r.NextDouble() * 0.8 - 0.4);
            rc[i] = (float)(0.05 + r.NextDouble() * 0.15); rb[i] = (float)(0.04 + r.NextDouble() * 0.06);
        }
    }

    static float Clamp01(float v) { return v < 0f ? 0f : (v > 1f ? 1f : v); }

    static int Scalar()
    {
        int hits = 0;
        float[] px = KernelBench.px, py = KernelBench.py, pz = KernelBench.pz, dx = KernelBench.dx, dy = KernelBench.dy, dz = KernelBench.dz;
        float[] qx = KernelBench.qx, qy = KernelBench.qy, qz = KernelBench.qz, ex = KernelBench.ex, ey = KernelBench.ey, ez = KernelBench.ez;
        float[] rc = KernelBench.rc, rb = KernelBench.rb; byte[] hitS = KernelBench.hitS; int n = px.Length;
        for (int i = 0; i < n; i++)
        {
            float rx = px[i] - qx[i], ry = py[i] - qy[i], rz = pz[i] - qz[i];
            float a = dx[i] * dx[i] + dy[i] * dy[i] + dz[i] * dz[i] + 1e-8f;
            float e = ex[i] * ex[i] + ey[i] * ey[i] + ez[i] * ez[i] + 1e-8f;
            float f = ex[i] * rx + ey[i] * ry + ez[i] * rz, c = dx[i] * rx + dy[i] * ry + dz[i] * rz;
            float b = dx[i] * ex[i] + dy[i] * ey[i] + dz[i] * ez[i];
            float den = a * e - b * b;
            float s = den > 1e-7f ? Clamp01((b * f - c * e) / den) : 0f;
            float t = Clamp01((b * s + f) / e);
            s = Clamp01((b * t - c) / a);
            float cx = rx + dx[i] * s - ex[i] * t, cy = ry + dy[i] * s - ey[i] * t, cz = rz + dz[i] * s - ez[i] * t;
            float rad = rc[i] + rb[i];
            byte h = (byte)(cx * cx + cy * cy + cz * cz <= rad * rad ? 1 : 0);
            hitS[i] = h; hits += h;
        }
        return hits;
    }

    static int Sphere()
    {
        int pass = 0;
        float[] px = KernelBench.px, py = KernelBench.py, pz = KernelBench.pz, dx = KernelBench.dx, dy = KernelBench.dy, dz = KernelBench.dz;
        float[] qx = KernelBench.qx, qy = KernelBench.qy, qz = KernelBench.qz, ex = KernelBench.ex, ey = KernelBench.ey, ez = KernelBench.ez;
        float[] rc = KernelBench.rc, rb = KernelBench.rb; int n = px.Length;
        for (int i = 0; i < n; i++)
        {
            float sx = px[i] + 0.5f * dx[i], sy = py[i] + 0.5f * dy[i], sz = pz[i] + 0.5f * dz[i];
            float cx = qx[i] + 0.5f * ex[i], cy = qy[i] + 0.5f * ey[i], cz = qz[i] + 0.5f * ez[i];
            float sr = 0.5f * (float)Math.Sqrt(dx[i] * dx[i] + dy[i] * dy[i] + dz[i] * dz[i]) + rb[i];
            float cr = 0.5f * (float)Math.Sqrt(ex[i] * ex[i] + ey[i] * ey[i] + ez[i] * ez[i]) + rc[i];
            float ddx = sx - cx, ddy = sy - cy, ddz = sz - cz, rr = sr + cr;
            pass += (ddx * ddx + ddy * ddy + ddz * ddz <= rr * rr) ? 1 : 0;
        }
        return pass;
    }

    static double Time(Func<int> fn, int reps, out double min, out double max)
    {
        for (int w = 0; w < 20; w++) fn();
        var samples = new double[9];
        for (int k = 0; k < 9; k++)
        {
            var sw = Stopwatch.StartNew();
            for (int r = 0; r < reps; r++) fn();
            samples[k] = sw.Elapsed.TotalMilliseconds * 1e6 / ((double)n * reps);
        }
        Array.Sort(samples);
        min = samples[0]; max = samples[8];
        return samples[4];
    }

    public static string Run(int count, int reps)
    {
        Fill(count);
        var sb = new StringBuilder();
        double mn, mx;
        sb.AppendLine(string.Format("runtime {0} | 64-bit {1} | tests per rep {2}", Environment.Version, Environment.Is64BitProcess, n));
        double sc = Time(Scalar, reps, out mn, out mx);
        sb.AppendLine(string.Format("scalar branchless float32 kernel : median {0:F2} ns/test (min {1:F2}, max {2:F2})", sc, mn, mx));
        double sp = Time(Sphere, reps, out mn, out mx);
        sb.AppendLine(string.Format("sphere prefilter scalar          : median {0:F2} ns/test (min {1:F2}, max {2:F2})", sp, mn, mx));
        sb.AppendLine(string.Format("hit fraction {0:F4}", Scalar() / (double)n));
        return sb.ToString();
    }
}
'@

$cp = New-Object System.CodeDom.Compiler.CompilerParameters
$cp.CompilerOptions = "/optimize+"
$cp.GenerateInMemory = $true
[void]$cp.ReferencedAssemblies.Add("System.dll")
Add-Type -TypeDefinition $code -Language CSharp -CompilerParameters $cp
$proc = [System.Diagnostics.Process]::GetCurrentProcess()
$proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::High
[KernelBench]::Run(65536, 60)
