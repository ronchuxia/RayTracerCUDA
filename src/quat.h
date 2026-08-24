#ifndef QUAT_H
#define QUAT_H

#include <cmath>

#include "precision.h"   // real
#include "vec3.h"

// Unit quaternion — the AUTHORITATIVE orientation of a rigid body (physics.h
// B3c). A rotation matrix cannot be integrated: adding omega*dt to nine numbers
// leaves a matrix that is no longer orthonormal, and renormalising it every step
// is both expensive and arbitrary. A quaternion has one constraint (unit length)
// instead of six, so integrate-then-normalise stays a rotation exactly.
//
// Convention: (x, y, z) is the vector part, w the scalar, and the matrix built
// below is the one whose COLUMNS are the body's own axes in world space —
// matching transforms.h's `apply_R(e_i)`, which is where box_collider_of reads a
// box's axes from. The two representations are checked against each other by
// tests/test_physics.cu.
struct quat {
    real x, y, z, w;

    __host__ __device__ quat() : x(0), y(0), z(0), w(1) {}    // identity
    __host__ __device__ quat(real _x, real _y, real _z, real _w)
        : x(_x), y(_y), z(_z), w(_w) {}
};

// Hamilton product: the rotation `b` followed by the rotation `a`.
__host__ __device__ inline quat operator*(const quat& a, const quat& b) {
    return quat(a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
                a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
                a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
                a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z);
}

// Renormalise. Called after every integration step: the linearised update below
// leaves the length slightly off, and the error compounds without this.
__host__ __device__ inline quat normalize(const quat& q) {
    real n = std::sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
    if (n <= real(0)) return quat();                 // degenerate -> identity
    real inv = real(1) / n;
    return quat(q.x*inv, q.y*inv, q.z*inv, q.w*inv);
}

// The body's i-th axis in world space — column i of the rotation matrix. This is
// what a box collider's `axes[]` cache holds, and what GJK's support function
// projects onto.
__host__ __device__ inline vec3 quat_axis(const quat& q, int i) {
    const real xx = q.x*q.x, yy = q.y*q.y, zz = q.z*q.z;
    const real xy = q.x*q.y, xz = q.x*q.z, yz = q.y*q.z;
    const real wx = q.w*q.x, wy = q.w*q.y, wz = q.w*q.z;
    if (i == 0) return vec3(real(1) - real(2)*(yy + zz),
                                       real(2)*(xy + wz),
                                       real(2)*(xz - wy));
    if (i == 1) return vec3(          real(2)*(xy - wz),
                            real(1) - real(2)*(xx + zz),
                                      real(2)*(yz + wx));
    return             vec3(          real(2)*(xz + wy),
                                      real(2)*(yz - wx),
                            real(1) - real(2)*(xx + yy));
}

// Rotation about `axis` (need not be unit) by `angle` radians.
__host__ __device__ inline quat quat_from_axis_angle(const vec3& axis, real angle) {
    real len = axis.length();
    if (len <= real(0)) return quat();
    real h = angle * real(0.5);
    real s = std::sin(h) / len;
    return quat(axis[0]*s, axis[1]*s, axis[2]*s, std::cos(h));
}

// Recover a quaternion from three orthonormal axes (the COLUMNS of R), so a body
// authored as a matrix — which is how transforms.h expresses a pose — can hand
// its orientation over without the caller knowing this convention.
//
// Shepperd's method: build from whichever of the four components is largest, so
// the divisor is never near zero. Taking the trace branch alone loses all
// precision at a 180-degree turn, where w -> 0.
__host__ __device__ inline quat quat_from_axes(const vec3 a[3]) {
    // m[row][col], with column c = a[c].
    const real m00 = a[0][0], m01 = a[1][0], m02 = a[2][0];
    const real m10 = a[0][1], m11 = a[1][1], m12 = a[2][1];
    const real m20 = a[0][2], m21 = a[1][2], m22 = a[2][2];
    const real tr = m00 + m11 + m22;
    if (tr > real(0)) {
        real s = std::sqrt(tr + real(1)) * real(2);          // s = 4w
        return normalize(quat((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s,
                              real(0.25) * s));
    }
    if (m00 > m11 && m00 > m22) {
        real s = std::sqrt(real(1) + m00 - m11 - m22) * real(2);   // s = 4x
        return normalize(quat(real(0.25) * s, (m01 + m10) / s, (m02 + m20) / s,
                              (m21 - m12) / s));
    }
    if (m11 > m22) {
        real s = std::sqrt(real(1) + m11 - m00 - m22) * real(2);   // s = 4y
        return normalize(quat((m01 + m10) / s, real(0.25) * s, (m12 + m21) / s,
                              (m02 - m20) / s));
    }
    real s = std::sqrt(real(1) + m22 - m00 - m11) * real(2);       // s = 4z
    return normalize(quat((m02 + m20) / s, (m12 + m21) / s, real(0.25) * s,
                          (m10 - m01) / s));
}

// One integration step of q under angular velocity `omega` (world frame,
// radians/s): dq/dt = 0.5 * omega_pure * q. First order, then renormalised —
// which is the standard choice for a fixed small step, and exactly what the
// unit-length constraint above is for.
__host__ __device__ inline quat quat_integrate(const quat& q, const vec3& omega, real dt) {
    const quat w(omega[0], omega[1], omega[2], real(0));
    const quat dq = w * q;
    const real h = real(0.5) * dt;
    return normalize(quat(q.x + dq.x*h, q.y + dq.y*h, q.z + dq.z*h, q.w + dq.w*h));
}

// Euler angles in DEGREES for transforms.h's convention, R = Rz(g)*Ry(b)*Rx(a)
// (Tait-Bryan ZYX). This is the COUPLING back to rendering: the physics owns a
// quaternion, the transform node owns Euler angles because that is what the
// viewer's editor fields show, and this is the one place they meet.
//
// Gimbal lock (the box pitched to exactly +/-90 degrees) collapses the x and z
// rotations into one degree of freedom; there the x angle is pinned to 0 and the
// whole turn is reported as z. The recovered matrix is still correct — only the
// split between the two angles is arbitrary, which is inherent to Euler angles
// and the reason the quaternion, not this, is authoritative.
__host__ __device__ inline vec3 quat_to_euler_zyx_degrees(const quat& q) {
    const vec3 c0 = quat_axis(q, 0), c1 = quat_axis(q, 1), c2 = quat_axis(q, 2);
    const real m00 = c0[0], m01 = c1[0];
    const real m10 = c0[1], m11 = c1[1];
    const real m20 = c0[2], m21 = c1[2], m22 = c2[2];
    real a, b, g;                                    // x, y, z rotations (radians)
    if (m20 <= real(-0.999999)) {                    // sin(b) = +1, b = +90 deg
        b = real(1.5707963267948966);
        a = real(0);
        g = -std::atan2(m01, m11);
    } else if (m20 >= real(0.999999)) {              // sin(b) = -1, b = -90 deg
        b = real(-1.5707963267948966);
        a = real(0);
        g = std::atan2(-m01, m11);
    } else {
        b = std::asin(-m20);
        a = std::atan2(m21, m22);
        g = std::atan2(m10, m00);
    }
    const real to_deg = real(57.295779513082320876798);
    return vec3(a * to_deg, b * to_deg, g * to_deg);
}

#endif // QUAT_H
