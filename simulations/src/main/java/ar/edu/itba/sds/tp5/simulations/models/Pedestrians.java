package ar.edu.itba.sds.tp5.simulations.models;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class Pedestrians implements Iterable<Particle>{

    private final double Ap = 1.1;
    private final double Bp = 2.1;
    private static final double EPS = 1e-12;

    private final List<Particle> particles;
    private double time;
    private final double L;
    private final int N;

    public Pedestrians(final int N, final double L, final double rMin, final double rMax,
                       final double vMax) {
        this.time = 0.0;
        this.L = L;
        this.N = N;
        //particles = generateParticles(N, L, rMin, rMax, vMax, null);
        particles = generateParticles(rMin, rMax, vMax);
    }

    private static double distPBC(MathVector p1, MathVector p2, double L) {
        double dx = p2.x() - p1.x();
        double dy = p2.y() - p1.y();
        dx -= L * Math.round(dx / L);
        dy -= L * Math.round(dy / L);
        return Math.hypot(dx, dy);
    }

//    public static List<Particle> generateParticles(final int N, final double L, final double rMin, final double rMax,
//        final double vMax, final Particle fixed) {
//
//        final List<Particle> ps = new ArrayList<>(N);
//
//        final int MAX_TRIES_PER_PARTICLE = 50_000;
//
//        for (int k = 0; k < N; k++) {
//            boolean placed = false;
//            for (int tries = 0; tries < MAX_TRIES_PER_PARTICLE; tries++) {
//                Particle cand = new Particle(rMin, rMax, vMax, L);
//
//                if (fixed != null) {
//                    double d0 = distPBC(cand.getPosition(), fixed.getPosition(), L);
//                    if (d0 <= (cand.getRMin() + fixed.getRMin() + EPS)) continue;
//                }
//
//                boolean ok = true;
//                for (Particle q : ps) {
//                    double d = distPBC(cand.getPosition(), q.getPosition(), L);
//
//                    if (d <= (cand.getR() + q.getR() + EPS)) { ok = false; break; }
//
//                    // No collition in rMax
//                    // if (d <= (rMax + rMax + EPS)) { ok = false; break; }
//                }
//
//                if (ok) {
//                    ps.add(cand);
//                    placed = true;
//                    break;
//                }
//            }
//            if (!placed) {
//                throw new IllegalStateException(
//                        "No pude ubicar la partícula #" + k + " sin solapes tras " + MAX_TRIES_PER_PARTICLE + " intentos."
//                );
//            }
//        }
//        return ps;
//    }

    //@TODO: borrar, es de prueba
    public static List<Particle> generateParticles(final double rMin, final double rMax, final double vMax) {
        return List.of(
                new Particle(1, 3, 5, 3, rMin, rMax, vMax),
                new Particle(5, 3, 1, 3, rMin, rMax, vMax)
        );
    }

    public void step(final double dt) {
        final int n = particles.size();
        final boolean[] inContact = new boolean[n];
        final MathVector[] dirThisStep = new MathVector[n];

        for (int i = 0; i < n; i++) {
            final Particle pi = particles.get(i);
            Particle best = null;
            double bestPen = Double.NEGATIVE_INFINITY;

            for (int j = 0; j < n; j++) {
                if (j == i) continue;
                final Particle pj = particles.get(j);
                if (!Contact.overlap(pi, pj, L)) continue;
                if (!Contact.contactAcp(pi, pj, L)) continue;

                final double dij = Contact.directDelta(pi.getPosition(), pj.getPosition()).length();
                final double pen = pi.getR() + pj.getR() - dij;
                if (pen > bestPen) { bestPen = pen; best = pj; }
            }

            if (best != null) {
                inContact[i] = true;
                dirThisStep[i] = Contact.escapeDir(pi, best, L);
            } else {
                inContact[i] = false;
                dirThisStep[i] = computeAvoidanceDirection(i);
            }

            if (dirThisStep[i] == null || dirThisStep[i].length() < EPS) {
                final MathVector vi = pi.getVelocity();
                dirThisStep[i] = (vi != null && vi.length() >= EPS) ? vi.normalize() : new MathVector(1.0, 0.0);
            }
        }

        for (int i = 0; i < n; i++) {
            final Particle p = particles.get(i);
            p.updateRadius(inContact[i], dt);
            p.updateVelocity(inContact[i], dirThisStep[i]);
        }

        for (int i = 0; i < n; i++) {
            final Particle p = particles.get(i);
            p.updatePosition(dt, L);
        }

        this.time += dt;
    }

    private MathVector  computeAvoidanceDirection(final int iIdx) {
        final Particle pi = particles.get(iIdx);
        final MathVector ri = pi.getPosition();
        MathVector et = pi.directionToTarget();
        if (et == null || et.length() < EPS) {
            MathVector vi = pi.getVelocity();
            et = (vi != null && vi.length() >= EPS) ? vi.normalize() : new MathVector(1.0, 0.0);
        } else {
            et = et.normalize();
        }

        MathVector heading = pi.getVelocity();
        heading = (heading != null && heading.length() >= EPS) ? heading.normalize() : et;

        final List<Integer> cand = new ArrayList<>();
        for (int j = 0; j < particles.size(); j++) {
            if (j == iIdx) continue;
            final MathVector rij = particles.get(j).getPosition().subtract(ri);
            final double rijLen = rij.length();
            if (rijLen < EPS) continue;

            double cosang = heading.dot(rij) / rijLen;
            cosang = clamp(cosang, -1.0, 1.0);
            double ang = Math.acos(cosang);
            if (ang <= Math.PI / 2.0 + 1e-12) {
                cand.add(j);
            }
        }
        cand.sort((a, b) -> {
            double da = particles.get(a).getPosition().subtract(ri).length();
            double db = particles.get(b).getPosition().subtract(ri).length();
            return Double.compare(da, db);
        });
        final int use = Math.min(2, cand.size());

        MathVector sum = MathVector.ZERO;
        final MathVector vi = (pi.getVelocity() != null) ? pi.getVelocity() : MathVector.ZERO;

        for (int k = 0; k < use; k++) {
            final Particle pj = particles.get(cand.get(k));

            MathVector eij = ri.subtract(pj.getPosition());
            final double dij = eij.length();
            if (dij < EPS) continue;
            eij = eij.scale(1.0 / dij);

            final MathVector vj = (pj.getVelocity() != null) ? pj.getVelocity() : MathVector.ZERO;
            final MathVector vij = vj.subtract(vi);
            if (vij.length() < EPS) continue;

            double denom = vij.length() * et.length();
            if (denom < EPS) continue;
            double cosb = vij.dot(et) / denom;
            cosb = clamp(cosb, -1.0, 1.0);
            double beta = Math.acos(cosb);
            if (beta < Math.PI / 2.0) {
                continue;
            }

            double cross = eij.x()*vij.y() - eij.y()*vij.x();
            double dot = eij.x()*vij.x() + eij.y()*vij.y();
            double alpha = Math.atan2(cross, dot);
            double sgn = Math.signum(alpha);
            // Case alpha ≈ 0 --> choose according to id
            if (Math.abs(cross) < 1e-12 && dot > 0) {
                sgn = (pi.getId() < pj.getId()) ? +1.0 : -1.0;
            }
            final double f = Math.abs(Math.abs(alpha) - Math.PI / 2.0);

            MathVector eij_c = eij.rotate(-sgn * f);
            MathVector njc = eij_c.scale(Ap * Math.exp(-dij / Bp));
            sum = sum.add(njc);
        }

        MathVector ea = et.add(sum);
        if (ea.length() < EPS) ea = et;
        else ea = ea.normalize();

        return ea;
    }

    private static double clamp(final double v, final double lo, final double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    @Override
    public Iterator<Particle> iterator() {
        return particles.iterator();
    }

    public List<Particle> getParticles() {
        return particles;
    }

    public double getTime() {
        return time;
    }
}
