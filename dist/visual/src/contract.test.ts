import { describe, expect, it } from 'vitest';
import { parseMotion, parsePrediction } from './data';
import { demoPrediction } from './demo';
import { clampRange, panView, zoomView } from './timelineMath';
import { interpolateOrientation, reconstructOrientation, validMapping } from './orientation';
import type { Imu, MotionManifest } from './data';

describe('visualization contract', () => {
  it('accepts timestamped model evidence while preserving canonical events', () => {
    expect(parsePrediction(demoPrediction).events).toEqual(demoPrediction.events);
    expect(() => parsePrediction({ ...demoPrediction, timeline: { ...demoPrediction.timeline, sessions: [{ ...demoPrediction.timeline!.sessions![0], macro: { timestamp_ms: [2, 1], probability: [.4, .5] } }] } })).toThrow('Invalid timeline stream');
  });
  it('reads the versioned binary record exactly', () => {
    const bytes = new ArrayBuffer(32), view = new DataView(bytes); view.setFloat64(0, 1234, true);
    [1, 2, 3, 4, 5, 6].forEach((v, i) => view.setFloat32(8 + i * 4, v, true));
    const manifest: MotionManifest = { telemetry_version:'1.0', session_id:'s', record_format:'f64_ms_6xf32_le', sample_count:1, start_ms:1234, end_ms:1234, binary_file:'motion.bin', units:{acceleration:'raw_adc',gyroscope:'raw_adc'} };
    const { imu } = parseMotion(manifest, bytes);
    expect([imu.t[0], imu.ax[0], imu.gz[0]]).toEqual([1234, 1, 6]);
    expect(() => parseMotion(manifest, new ArrayBuffer(31))).toThrow();
  });
  it('clamps selection and synchronized viewport limits', () => {
    const bounds = {start_ms:0,end_ms:1000};
    expect(clampRange(1200,-100,bounds)).toEqual(bounds);
    expect(zoomView(bounds,900,200)).toEqual({start_ms:800,end_ms:1000});
    expect(panView(bounds,{start_ms:800,end_ms:1000},500)).toEqual({start_ms:800,end_ms:1000});
  });
});

describe('IMU orientation', () => {
  it('rejects reflected or uncalibrated coordinate frames', () => {
    expect(validMapping([1,0,0,0,1,0,0,0,-1])).toBe(false);
    expect(validMapping([1,0,0,0,0,1,0,-1,0])).toBe(true);
  });
  it('integrates a known rotation, interpolates, and refuses to bridge gaps', () => {
    const imu: Imu = { t:new Float64Array([0,100,200,900]), ax:new Float32Array(4), ay:new Float32Array([1,1,1,1]), az:new Float32Array(4), gx:new Float32Array(4), gy:new Float32Array(4), gz:new Float32Array([0,Math.PI/2,Math.PI/2,0]) };
    const m: MotionManifest = { telemetry_version:'1.0',session_id:'s',record_format:'f64_ms_6xf32_le',sample_count:4,start_ms:0,end_ms:900,binary_file:'motion.bin',units:{acceleration:'g',gyroscope:'rad/s'},calibration:{acceleration_counts_per_g:1,gyroscope_counts_per_rad_s:1,viewer_from_sensor:[1,0,0,0,1,0,0,0,1]} };
    const points = reconstructOrientation(imu,m)!;
    expect(points[2].z).toBeGreaterThan(.13);
    expect(interpolateOrientation(points,150)).not.toBeNull();
    expect(interpolateOrientation(points,500)).toBeNull();
    expect(reconstructOrientation(imu,{...m,calibration:undefined})).toBeNull();
  });
});
