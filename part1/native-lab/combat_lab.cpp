// Experimental standalone server lab. Synthetic asset trajectories;
// this is NOT a production MMO, engine integration or human playtest.
// Build with build.cmd. See run_lab.py and generated JSON for reproducibility.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define FD_SETSIZE 512
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>
using Clock=std::chrono::steady_clock;
constexpr double DT=1.0/30.0, EPS=1e-6, PI=3.141592653589793;
double millis(Clock::time_point p){return std::chrono::duration<double,std::milli>(Clock::now()-p).count();}
struct V{double x=0,y=0,z=0; V operator+(V b)const{return{x+b.x,y+b.y,z+b.z};} V operator-(V b)const{return{x-b.x,y-b.y,z-b.z};} V operator*(double s)const{return{x*s,y*s,z*s};}};
double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;} double norm(V a){return std::sqrt(dot(a,a));}
V lerp(V a,V b,double t){return a+(b-a)*t;}
struct Capsule{V a,b;double r;};
struct Motion{Capsule p,q;}; // endpoints affine in normalized [0,1] interval
Capsule at(const Motion&m,double t){return{lerp(m.p.a,m.q.a,t),lerp(m.p.b,m.q.b,t),m.p.r};}
double pointDist(V p,V a,V b){V d=b-a;double t=dot(d,d)>1e-24?std::clamp(dot(p-a,d)/dot(d,d),0.,1.):0;V v=p-(a+d*t);return dot(v,v);}
// Segment distance: closest unconstrained point when valid, plus four boundaries.
// Handles zero-length capsules (spheres) and parallel segments explicitly.
double distance2(Capsule a,Capsule b){
 V u=a.b-a.a,v=b.b-b.a,w=a.a-b.a;double A=dot(u,u),B=dot(u,v),C=dot(v,v),D=dot(u,w),E=dot(v,w),den=A*C-B*B;
 double best=std::min({pointDist(a.a,b.a,b.b),pointDist(a.b,b.a,b.b),pointDist(b.a,a.a,a.b),pointDist(b.b,a.a,a.b)});
 if(den>1e-14*A*C&&A>1e-24&&C>1e-24){double s=(B*E-C*D)/den,t=(A*E-B*D)/den;if(s>=0&&s<=1&&t>=0&&t<=1){V d=w+u*s-v*t;best=std::min(best,dot(d,d));}}
 return best;
}
struct Box{V lo,hi;};
Box bounds(Motion m){double r=m.p.r;return{{std::min({m.p.a.x,m.p.b.x,m.q.a.x,m.q.b.x})-r,std::min({m.p.a.y,m.p.b.y,m.q.a.y,m.q.b.y})-r,std::min({m.p.a.z,m.p.b.z,m.q.a.z,m.q.b.z})-r},{std::max({m.p.a.x,m.p.b.x,m.q.a.x,m.q.b.x})+r,std::max({m.p.a.y,m.p.b.y,m.q.a.y,m.q.b.y})+r,std::max({m.p.a.z,m.p.b.z,m.q.a.z,m.q.b.z})+r}};}
Box unite(Box a,Box b){return{{std::min(a.lo.x,b.lo.x),std::min(a.lo.y,b.lo.y),std::min(a.lo.z,b.lo.z)},{std::max(a.hi.x,b.hi.x),std::max(a.hi.y,b.hi.y),std::max(a.hi.z,b.hi.z)}};}
bool overlaps(Box a,Box b){return a.lo.x<=b.hi.x&&a.hi.x>=b.lo.x&&a.lo.y<=b.hi.y&&a.hi.y>=b.lo.y&&a.lo.z<=b.hi.z&&a.hi.z>=b.lo.z;}
struct Contact{bool hit=false;double t=1;int eval=0;bool unresolved=false;};
// Lipschitz-safe advancement: max endpoint displacement bounds speed of every
// point on an affine capsule. The sum bounds separation's rate of decrease.
// A contact can be up to EPS metres early spatially. Never silently discard an
// unresolved interval after the iteration cap; flag it and use interval fallback.
Contact ccd(Motion a,Motion b){
 double speed=std::max(norm(a.q.a-a.p.a),norm(a.q.b-a.p.b))+std::max(norm(b.q.a-b.p.a),norm(b.q.b-b.p.b));
 double r=a.p.r+b.p.r,t=0;int n=0;
 for(;n<128;n++){
  double gap=std::sqrt(distance2(at(a,t),at(b,t)))-r;
  if(gap<=EPS)return{true,t,n+1,false};
  if(speed<1e-15||t>=1)return{false,1,n+1,false};
  double step=gap/speed;
  if(t+step>1){double end=std::sqrt(distance2(a.q,b.q))-r;return{end<=EPS,1,n+2,false};}
  t+=step*0.99;
 }
 struct I{double a,b;int depth;};std::vector<I> todo{{t,1,0}};
 while(!todo.empty()){
  I i=todo.back();todo.pop_back();double mid=(i.a+i.b)*.5,gap=std::sqrt(distance2(at(a,mid),at(b,mid)))-r;n++;
  // Prove separation from the actual surface, not from an EPS-expanded shell.
  // Otherwise intervals just outside EPS recurse forever at the shell edge.
  if(gap>speed*(i.b-i.a)*.5)continue;
  if(speed*(i.b-i.a)<=EPS&&gap<=EPS)return{true,mid,n,false};
  if(i.depth>=50||n>20000)return{false,mid,n,true};
  todo.push_back({mid,i.b,i.depth+1});todo.push_back({i.a,mid,i.depth+1});
 }
 return{false,1,n,false};
}
double percentile(std::vector<double> v,double p){if(v.empty())return 0;std::sort(v.begin(),v.end());return v[std::min(v.size()-1,(size_t)std::ceil(p*v.size())-1)];}
void stats(const char*n,const std::vector<double>&v){std::cout<<"\""<<n<<"\":{\"p50\":"<<percentile(v,.5)<<",\"p95\":"<<percentile(v,.95)<<",\"p99\":"<<percentile(v,.99)<<",\"p999\":"<<percentile(v,.999)<<",\"max\":"<<(v.empty()?0:*std::max_element(v.begin(),v.end()))<<"}";}
struct Hit{int a,b;double t;};
struct Fixture{
 int N,H,S,W=3,P=1000;double spacing;bool prune,networkInput=false;std::vector<V> roots,old;std::array<V,200> desired{};std::vector<double> hp,guard;
 std::vector<Motion> hurt,weapons;std::vector<Box> hurtBox,weaponBox,rootBox,attackBox;
 std::vector<Hit> hits;std::vector<int> seen;std::vector<V> bolts,boltVel;std::vector<std::array<V,200>> history;
 uint64_t candidates=0,tests=0,evaluations=0,uncertain=0,aoe=0,projectileHits=0,blocks=0,dodges=0;double lastPen=0;
 Fixture(int n,int h,int s,double sp,bool pr):N(n),H(h),S(s),spacing(sp),prune(pr),roots(n),old(n),hp(n,1e9),guard(n),hurt(n*h*s),weapons(n*W*s),hurtBox(n*h*s),weaponBox(n*W*s),rootBox(n),attackBox(n),seen(n*n,-1),bolts(P),boltVel(P),history(16){
  for(int i=0;i<N;i++)roots[i]={(i%20-9.5)*spacing,(i/20-4.5)*spacing,0};old=roots;hits.reserve(n*n);
  for(int i=0;i<P;i++){bolts[i]={(i%31-15)*.3,(i%37-18)*.3,1.0+(i%7)*.1};boltVel[i]={10*std::cos(i),10*std::sin(i),0};}
 }
 Capsule body(int i,int h,double f,int tick){
  // Eight articulated synthetic capsules, split into H/8 subcapsules without
  // altering their union. This allows varying hitbox count with same geometry.
  static const Capsule base[8]={{{0,0,1.55},{0,0,1.7},.14},{{0,0,.95},{0,0,1.5},.2},{{-.2,0,1.4},{-.45,0,1},.1},{{.2,0,1.4},{.45,0,1},.1},{{-.15,0,.15},{-.15,0,.9},.12},{{.15,0,.15},{.15,0,.9},.12},{{-.45,0,.95},{-.55,0,.9},.08},{{.45,0,.95},{.55,0,.9},.08}};
  int parts=H/8,k=h/parts,part=h%parts;Capsule c=base[k];V d=c.b-c.a;c.b=c.a+d*((part+1.)/parts);c.a=c.a+d*(double(part)/parts);
  double yaw=.15*std::sin((tick+f)*.13+i*.3),cs=std::cos(yaw),sn=std::sin(yaw);V root=lerp(old[i],roots[i],f);
  auto tr=[&](V v){return root+V{cs*v.x-sn*v.y,sn*v.x+cs*v.y,v.z+.03*std::sin((tick+f)*.2+i)};};return{tr(c.a),tr(c.b),c.r};
 }
 Capsule blade(int i,int w,double f,int tick){
  double yaw=i*.77+(tick%4+f)*(PI/4),r0=.4+w*.55,r1=r0+.55;V r=lerp(old[i],roots[i],f);
  return{r+V{r0*std::cos(yaw),r0*std::sin(yaw),1.1},r+V{r1*std::cos(yaw),r1*std::sin(yaw),1.1},.065};
 }
 void crowd(int tick){
  old=roots;for(int i=0;i<N;i++){double a=.5*std::sin(i*.7+tick*.04);V velocity=networkInput?desired[i]:V{a,.5*std::cos(i*.3+tick*.03),0};roots[i]=roots[i]+velocity*DT;}
  // Experiment policy: symmetric hard body discs, projected Gauss-Seidel.
  // Residual penetration is measured, not assumed zero.
  std::vector<V> corr(N);
  for(int it=0;it<64;it++){double residual=0;for(int i=0;i<N;i++)for(int j=i+1;j<N;j++){
   V d=roots[j]-roots[i];double ds=dot(d,d);if(ds>=.7*.7)continue;double len=std::sqrt(ds);residual=std::max(residual,.7-len);V normal=len>1e-9?d*(1/len):V{1,0,0};V c=normal*((.7-len)*.5);roots[i]=roots[i]-c;roots[j]=roots[j]+c;
  }for(int i=0;i<N;i++){roots[i].x=std::clamp(roots[i].x,-10.,10.);roots[i].y=std::clamp(roots[i].y,-6.,6.);}if(residual<1e-5)break;}
  lastPen=0;for(int i=0;i<N;i++)for(int j=i+1;j<N;j++)lastPen=std::max(lastPen,.7-norm(roots[j]-roots[i]));
  for(int i=0;i<N;i++)history[tick%16][i]=roots[i];
 }
 void pose(int tick){
  for(int i=0;i<N;i++){
   bool first=true;for(int s=0;s<S;s++)for(int h=0;h<H;h++){int k=(i*S+s)*H+h;hurt[k]={body(i,h,double(s)/S,tick),body(i,h,double(s+1)/S,tick)};hurtBox[k]=bounds(hurt[k]);rootBox[i]=first?hurtBox[k]:unite(rootBox[i],hurtBox[k]);first=false;}
   first=true;for(int s=0;s<S;s++)for(int w=0;w<W;w++){int k=(i*S+s)*W+w;weapons[k]={blade(i,w,double(s)/S,tick),blade(i,w,double(s+1)/S,tick)};weaponBox[k]=bounds(weapons[k]);attackBox[i]=first?weaponBox[k]:unite(attackBox[i],weaponBox[k]);first=false;}
  }
 }
 void melee(int tick){
  hits.clear();candidates=tests=evaluations=uncertain=0;
  for(int i=0;i<N;i++)for(int j=0;j<N;j++){
   if(i==j)continue;if(prune&&!overlaps(attackBox[i],rootBox[j]))continue;candidates++;
   double earliest=2;bool any=false;
   for(int s=0;s<S;s++){
    if(any&&double(s)/S>earliest)break;
    for(int w=0;w<W;w++)for(int h=0;h<H;h++){
     int a=(i*S+s)*W+w,b=(j*S+s)*H+h;if(prune&&!overlaps(weaponBox[a],hurtBox[b]))continue;tests++;
     auto c=ccd(weapons[a],hurt[b]);evaluations+=c.eval;if(c.unresolved)uncertain++;
     if(c.hit){any=true;earliest=std::min(earliest,(s+c.t)/S);}
    }
   }
   if(any)hits.push_back({i,j,earliest});
  }
 }
 void effects(int tick){
  // Snapshot defense conditions at contact sub-time. Rules below are test
  // fixtures, not project design decisions. Each attack has a four-tick window.
  for(auto h:hits){int key=h.a*N+h.b,attack=tick/4;if(seen[key]==attack)continue;seen[key]=attack;
   double phase=std::fmod(tick+h.t+h.b*.1,20.);if(phase<2){dodges++;continue;}
   V dir=roots[h.a]-roots[h.b];bool front=dir.x*std::cos(h.b*.77)+dir.y*std::sin(h.b*.77)>0;
   if(phase<6&&front){blocks++;continue;}hp[h.b]-=10;
  }
  // Worst burst every tick, uncapped, per-source/target resistance arithmetic.
  aoe=0;for(int i=0;i<N;i++)for(int j=0;j<N;j++)if(i!=j){double d=norm(roots[j]-roots[i]);if(d<30){hp[j]-=.1*(1+(i%5)*.1)/(1+(j%7)*.1);aoe++;}}
  for(int p=0;p<P;p++){
   V next=bolts[p]+boltVel[p]*DT;Motion m{{bolts[p],bolts[p],.025},{next,next,.025}};Box bb=bounds(m);double best=2;int who=-1;
   for(int j=0;j<N;j++){if(!overlaps(bb,rootBox[j]))continue;for(int s=0;s<S;s++)for(int h=0;h<H;h++){
    Motion part{{lerp(bolts[p],next,double(s)/S),lerp(bolts[p],next,double(s)/S),.025},{lerp(bolts[p],next,double(s+1)/S),lerp(bolts[p],next,double(s+1)/S),.025}};
    int k=(j*S+s)*H+h;if(!overlaps(bounds(part),hurtBox[k]))continue;auto c=ccd(part,hurt[k]);if(c.unresolved)uncertain++;if(c.hit&&(s+c.t)/S<best){best=(s+c.t)/S;who=j;}
   }}
   if(who>=0){hp[who]-=3;projectileHits++;}
   if(who>=0||norm(next)>15){bolts[p]={10*std::cos(p+tick*.01),5*std::sin(p+tick*.01),1.1};}else bolts[p]=next;
  }
 }
};
void verify(){
 std::mt19937_64 rng(20260916);std::uniform_real_distribution<double> u(-3,3);int mismatch=0,unc=0;double maxerr=0;
 // Independent analytical closest approach and quadratic TOI for translating
 // spheres. Includes initial overlap, zero relative speed, tangent, crossing.
 for(int k=0;k<20000;k++){
  V p{u(rng),u(rng),u(rng)},q{u(rng),u(rng),u(rng)},v{u(rng),u(rng),u(rng)},w{u(rng),u(rng),u(rng)};
  if(k%10==0)w=v;if(k%13==0)q=p;double r=.4;
  Motion a{{p,p,r/2},{p+v,p+v,r/2}},b{{q,q,r/2},{q+w,q+w,r/2}};
  V d=p-q,dv=v-w;double A=dot(dv,dv),t=A>1e-20?std::clamp(-dot(d,dv)/A,0.,1.):0;double gap=norm(d+dv*t)-r;bool expected=gap<=0;
  auto c=ccd(a,b);unc+=c.unresolved;if(c.hit!=expected&&std::abs(gap)>2*EPS)mismatch++;
  if(expected&&dot(d,d)>r*r&&A>1e-20){double B=dot(d,dv),C=dot(d,d)-r*r,D=B*B-A*C;if(D>=0){double exact=(-B-std::sqrt(D))/A;maxerr=std::max(maxerr,std::abs(c.t-exact));}}
 }
 // Broadphase optimization checked against unfiltered continuous-capsule solver.
 int setdiff=0;double toiDiff=0;Fixture f(200,16,4,.75,true);
 for(int t=0;t<4;t++){f.crowd(t);f.pose(t);f.prune=true;f.melee(t);auto fast=f.hits;f.prune=false;f.melee(t);auto slow=f.hits;unc+=f.uncertain;
  if(fast.size()!=slow.size())setdiff+=int(std::abs((long long)fast.size()-(long long)slow.size()));
  else for(size_t k=0;k<fast.size();k++){if(fast[k].a!=slow[k].a||fast[k].b!=slow[k].b)setdiff++;toiDiff=std::max(toiDiff,std::abs(fast[k].t-slow[k].t));}
 }
 std::cout<<"{\"sphere_cases\":20000,\"sphere_mismatches\":"<<mismatch<<",\"unresolved\":"<<unc<<",\"max_sphere_toi_fraction_error\":"<<maxerr<<",\"broadphase_set_mismatches\":"<<setdiff<<",\"broadphase_toi_error\":"<<toiDiff<<"}\n";
 if(mismatch||setdiff||unc)std::exit(2);
}
void bench(int ticks,int h,int sub,bool prune,double spacing){
 Fixture f(200,h,sub,spacing,prune);std::vector<double> all,crowd,pose,hit,effect;uint64_t candidates=0,tests=0,ev=0,unresolved=0,contacts=0;double pen=0;
 for(int t=0;t<ticks+120;t++){
  auto a=Clock::now();f.crowd(t);double c=millis(a);auto b=Clock::now();f.pose(t);double p=millis(b);b=Clock::now();f.melee(t);double m=millis(b);b=Clock::now();f.effects(t);double e=millis(b);double total=millis(a);
  if(t>=120){all.push_back(total);crowd.push_back(c);pose.push_back(p);hit.push_back(m);effect.push_back(e);candidates+=f.candidates;tests+=f.tests;ev+=f.evaluations;unresolved+=f.uncertain;contacts+=f.hits.size();pen=std::max(pen,f.lastPen);}
 }
 std::cout<<"{\"players\":200,\"hurtboxes\":"<<h<<",\"subintervals\":"<<sub<<",\"pruning\":"<<(prune?"true":"false")<<",\"spacing\":"<<spacing<<",\"ticks\":"<<ticks<<",";stats("tick_ms",all);std::cout<<",";stats("crowd_ms",crowd);std::cout<<",";stats("pose_ms",pose);std::cout<<",";stats("melee_ms",hit);std::cout<<",";stats("effects_projectiles_ms",effect);
 std::cout<<",\"mean_candidates\":"<<double(candidates)/ticks<<",\"mean_ccd_tests\":"<<double(tests)/ticks<<",\"mean_distance_evals\":"<<double(ev)/ticks<<",\"contacts\":"<<contacts<<",\"unresolved\":"<<unresolved<<",\"max_penetration_m\":"<<pen<<",\"aoe_apps_per_tick\":"<<f.aoe<<",\"projectile_hits\":"<<f.projectileHits<<",\"blocks\":"<<f.blocks<<",\"dodges\":"<<f.dodges<<",\"checksum\":"<<std::accumulate(f.hp.begin(),f.hp.end(),0.)<<"}\n";
 if(unresolved)std::exit(2);
}
// Independent static reference: minimize convex point-to-segment distance
// along the other segment using ternary search, without the pair solver formula.
double referenceDistance(Capsule a,Capsule b){double lo=0,hi=1;for(int k=0;k<100;k++){double x=(2*lo+hi)/3,y=(lo+2*hi)/3;if(pointDist(lerp(a.a,a.b,x),b.a,b.b)<pointDist(lerp(a.a,a.b,y),b.a,b.b))hi=y;else lo=x;}return pointDist(lerp(a.a,a.b,(lo+hi)/2),b.a,b.b);}
void verifyExtended(){
 std::mt19937_64 rng(611930);std::uniform_real_distribution<double> u(-2,2);auto v=[&](){return V{u(rng),u(rng),u(rng)};};int distanceBad=0,falseNegative=0,invalidContact=0,unc=0;double maxDistanceError=0;
 for(int i=0;i<30000;i++){Capsule a{v(),v(),.1},b{v(),v(),.1};if(i%7==0)a.b=a.a;if(i%11==0)b.b=b.a+(a.b-a.a);if(i%13==0)b.a=a.a;double err=std::abs(distance2(a,b)-referenceDistance(a,b));maxDistanceError=std::max(maxDistanceError,err);if(err>1e-9)distanceBad++;}
 // Dense independent TIME samples are witnesses of overlap, not a proof that
 // misses are absent between all samples. Analytic speed bound supplies that.
 for(int i=0;i<4000;i++){Motion a{{v(),v(),.08},{v(),v(),.08}},b{{v(),v(),.12},{v(),v(),.12}};auto c=ccd(a,b);unc+=c.unresolved;bool sampled=false;for(int k=0;k<=2048;k++)if(distance2(at(a,k/2048.),at(b,k/2048.))<=.2*.2){sampled=true;break;}if(sampled&&!c.hit)falseNegative++;if(c.hit&&std::sqrt(referenceDistance(at(a,c.t),at(b,c.t)))-.2>EPS*1.01)invalidContact++;}
 // History is indexed by simulation tick, including bounds and defense state.
 // A target moving out of current bounds must still be found at the rewind tick.
 std::array<Motion,16> history;for(int t=0;t<16;t++){double x=t<8?1.:10.;history[t]={{{x,0,0},{x,0,1},.2},{{x,0,0},{x,0,1},.2}};}Motion sword{{{0,0,.5},{2,0,.5},.05},{{0,0,.5},{2,0,.5},.05}};bool historical=overlaps(bounds(sword),bounds(history[5]))&&ccd(sword,history[5]).hit;bool current=overlaps(bounds(sword),bounds(history[15]));int historicalFailure=(!historical||current)?1:0;
 std::cout<<"{\"independent_static_cases\":30000,\"distance_mismatches\":"<<distanceBad<<",\"max_squared_distance_error\":"<<maxDistanceError<<",\"moving_capsule_cases\":4000,\"samples_per_case\":2049,\"sample_witness_misses\":"<<falseNegative<<",\"invalid_reported_contacts\":"<<invalidContact<<",\"unresolved\":"<<unc<<",\"historical_broadphase_regression_failures\":"<<historicalFailure<<"}\n";if(distanceBad||falseNegative||invalidContact||unc||historicalFailure)std::exit(2);
}
#include "network_lab.h"
int main(int argc,char**argv){std::cout<<std::setprecision(10);std::string mode=argc>1?argv[1]:"verify";if(mode=="verify")verify();else if(mode=="verifyextended")verifyExtended();else if(mode=="bench")bench(argc>2?atoi(argv[2]):600,argc>3?atoi(argv[3]):16,argc>4?atoi(argv[4]):4,argc>5?atoi(argv[5])!=0:true,argc>6?atof(argv[6]):.75);else if(mode=="server")networkServer(argc>2?atoi(argv[2]):900,argc>3?atoi(argv[3]):39091);else if(mode=="clients")networkClients(argc>2?atoi(argv[2]):900,argc>3?atoi(argv[3]):39091);else return 1;}
