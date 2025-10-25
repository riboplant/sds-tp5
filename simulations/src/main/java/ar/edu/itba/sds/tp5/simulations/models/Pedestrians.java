package ar.edu.itba.sds.tp5.simulations.models;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class Pedestrians implements Iterable<Particle>{

    private static final double EPS = 1e-6;

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
        this.time += dt;
        final int n = particles.size();

        boolean[] inContact = new boolean[n];
        final MathVector[] dirThisStep = new MathVector[n];

        for (int id = 0; id < n; id++) {
            Particle p = particles.get(id);

            Particle best = null;
            double bestPenetration = Double.NEGATIVE_INFINITY;

            for (int id2 = 0; id2 < n; id2++) {
                if (id == id2) continue;
                Particle q = particles.get(id2);
                if (!Contact.overlap(p, q, L) || !Contact.contactAcp(p, q, L)) continue;
                double dij = Contact.directDelta(p.getPosition(), q.getPosition()).length();
                double penetration = p.getR() + q.getR() - dij;
                if (penetration > bestPenetration) {
                    bestPenetration = penetration;
                    best = q;
                }
            }

            if (best != null) {
                inContact[id] = true;
                dirThisStep[id]= Contact.escapeDir(p, best, L);
            } else {
                inContact[id] = false;
                MathVector eT = p.directionToTarget();
                double nE = eT.length();
                dirThisStep[id] = (nE < 1e-12) ? MathVector.ZERO : eT.scale(1.0/nE);
            }
        }

        for (int id = 0; id < n; id++) {
            final Particle p = particles.get(id);
            p.updateRadius(inContact[id], dt);
            p.updateVelocity(inContact[id], dirThisStep[id]);
        }

        for (int id = 0; id < n; id++) {
            final Particle p = particles.get(id);
            p.updatePosition(dt);
        }

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
