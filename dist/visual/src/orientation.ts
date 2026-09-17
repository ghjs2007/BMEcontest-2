import type { Imu, MotionManifest } from './data';
export type Quaternion = { x: number; y: number; z: number; w: number };
export type QuaternionPoint = Quaternion & { t: number; segment: number };
const identity: Quaternion = { x: 0, y: 0, z: 0, w: 1 };
export function normalize(q: Quaternion): Quaternion { const n = Math.hypot(q.x,q.y,q.z,q.w); return n > 1e-12 && Number.isFinite(n) ? {x:q.x/n,y:q.y/n,z:q.z/n,w:q.w/n} : identity; }
export function multiply(a: Quaternion,b: Quaternion): Quaternion { return normalize({ x:a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,y:a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,z:a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w,w:a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z }); }
export function fromAxisAngle(x: number,y: number,z: number,angle: number): Quaternion { const n=Math.hypot(x,y,z); if(n<1e-12)return identity; const s=Math.sin(angle/2)/n; return normalize({x:x*s,y:y*s,z:z*s,w:Math.cos(angle/2)}); }
export function rotate(q: Quaternion,v: [number,number,number]): [number,number,number] { const [x,y,z]=v; const qv={x,y,z,w:0}; const inv={x:-q.x,y:-q.y,z:-q.z,w:q.w}; const a=rawMultiply(rawMultiply(q,qv),inv); return [a.x,a.y,a.z]; }
function rawMultiply(a:Quaternion,b:Quaternion):Quaternion{return{x:a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,y:a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,z:a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w,w:a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z};}
export function slerp(a: Quaternion,b: Quaternion,alpha:number):Quaternion { let dot=a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w,bb=b; if(dot<0){dot=-dot;bb={x:-b.x,y:-b.y,z:-b.z,w:-b.w};} if(dot>.9995)return normalize({x:a.x+(bb.x-a.x)*alpha,y:a.y+(bb.y-a.y)*alpha,z:a.z+(bb.z-a.z)*alpha,w:a.w+(bb.w-a.w)*alpha}); const theta=Math.acos(Math.max(-1,Math.min(1,dot))),s=Math.sin(theta); return normalize({x:(a.x*Math.sin((1-alpha)*theta)+bb.x*Math.sin(alpha*theta))/s,y:(a.y*Math.sin((1-alpha)*theta)+bb.y*Math.sin(alpha*theta))/s,z:(a.z*Math.sin((1-alpha)*theta)+bb.z*Math.sin(alpha*theta))/s,w:(a.w*Math.sin((1-alpha)*theta)+bb.w*Math.sin(alpha*theta))/s}); }
export function validMapping(matrix: number[]): boolean {
  if (!Array.isArray(matrix)||matrix.length!==9||matrix.some(v=>!Number.isFinite(v)))return false;
  const rows=[matrix.slice(0,3),matrix.slice(3,6),matrix.slice(6,9)];
  const determinant = matrix[0]*(matrix[4]*matrix[8]-matrix[5]*matrix[7])-matrix[1]*(matrix[3]*matrix[8]-matrix[5]*matrix[6])+matrix[2]*(matrix[3]*matrix[7]-matrix[4]*matrix[6]);
  return rows.every(r=>Math.abs(Math.hypot(...r)-1)<1e-5) && Math.abs(rows[0].reduce((s,x,i)=>s+x*rows[1][i],0))<1e-5 && Math.abs(rows[0].reduce((s,x,i)=>s+x*rows[2][i],0))<1e-5 && Math.abs(rows[1].reduce((s,x,i)=>s+x*rows[2][i],0))<1e-5 && Math.abs(determinant-1)<1e-5;
}
export function canOrient(m: MotionManifest): boolean {
  const c=m.calibration;
  return !!c && validMapping(c.viewer_from_sensor) && (m.units.acceleration==='g'||c.acceleration_counts_per_g>0) && (m.units.gyroscope==='rad/s'||m.units.gyroscope==='deg/s'||c.gyroscope_counts_per_rad_s>0);
}
function map(v:[number,number,number],m:number[]):[number,number,number]{return[m[0]*v[0]+m[1]*v[1]+m[2]*v[2],m[3]*v[0]+m[4]*v[1]+m[5]*v[2],m[6]*v[0]+m[7]*v[1]+m[8]*v[2]];}
function gravityTilt(up:[number,number,number]):Quaternion { const n=Math.hypot(...up); if(n<1e-8)return identity; const u=up.map(x=>x/n) as [number,number,number]; const axis:[number,number,number]=[-u[2],0,u[0]]; const dot=Math.max(-1,Math.min(1,u[1])); return fromAxisAngle(...axis,Math.acos(dot)); }
export function reconstructOrientation(imu: Imu,m: MotionManifest):QuaternionPoint[] | null {
  if(!canOrient(m))return null;
  const c=m.calibration!,basis=c.viewer_from_sensor,points:QuaternionPoint[]=[];
  let q=identity,last=-Infinity,lastStored=-Infinity,segment=0;
  for(let i=0;i<imu.t.length;i++){
    const time=imu.t[i],dt=(time-last)/1000;
    if(!Number.isFinite(time)||time<last)throw new Error('Unordered IMU timestamps.');
    if(time===last)continue;
    const accFactor=m.units.acceleration==='g'?1:1/c.acceleration_counts_per_g;
    const gyroFactor=m.units.gyroscope==='rad/s'?1:m.units.gyroscope==='deg/s'?Math.PI/180:1/c.gyroscope_counts_per_rad_s;
    const acc=map([imu.ax[i]*accFactor,imu.ay[i]*accFactor,imu.az[i]*accFactor],basis);
    const gyro=map([imu.gx[i]*gyroFactor,imu.gy[i]*gyroFactor,imu.gz[i]*gyroFactor],basis);
    if(!acc.every(Number.isFinite)||!gyro.every(Number.isFinite)){last=time;continue;}
    const norm=Math.hypot(...acc);
    if(!Number.isFinite(dt)||dt>.5){if(last!==-Infinity)segment++;q=norm>.4&&norm<1.6?gravityTilt(acc):identity;}
    else if(dt>0){const omega=Math.hypot(...gyro);q=multiply(q,fromAxisAngle(...gyro,omega*dt));
      if(norm>.8&&norm<1.2){const up=rotate(q,[acc[0]/norm,acc[1]/norm,acc[2]/norm]);const error:[number,number,number]=[-up[2],0,up[0]];const e=Math.hypot(...error);if(e>1e-8)q=multiply(fromAxisAngle(...error,Math.min(.025,e*.025)),q);}
    }
    last=time;
    if(time-lastStored>=40||points.length===0){points.push({t:time,...normalize(q),segment});lastStored=time;}
  }
  return points;
}
export function interpolateOrientation(points:QuaternionPoint[],time:number):Quaternion|null {
  if(!points.length||time<points[0].t||time>points[points.length-1].t)return null;
  let lo=0,hi=points.length-1;while(lo<hi){const mid=Math.floor((lo+hi)/2);if(points[mid].t<time)lo=mid+1;else hi=mid;}
  if(lo===0)return points[0];const a=points[lo-1],b=points[lo];if(a.segment!==b.segment||b.t-a.t>500)return null;
  return slerp(a,b,(time-a.t)/(b.t-a.t));
}
