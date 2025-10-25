package ar.edu.itba.sds.tp5.simulations.models;

public final class Contact {

    private static final double EPS = 1e-12;
    private Contact(){}

    public static MathVector minImage(MathVector from, MathVector to, double L) {
        double dx = to.x() - from.x();
        double dy = to.y() - from.y();
        dx -= L * Math.round(dx / L);
        dy -= L * Math.round(dy / L);
        return new MathVector(dx, dy);
    }

    public static MathVector directDelta(MathVector from, MathVector to) {
        return new MathVector(to.x() - from.x(), to.y() - from.y());
    }

    // @TODO chequear
    public static boolean isAhead(Particle i, Particle j, double L) {
        MathVector rij = directDelta(i.getPosition(), j.getPosition());
        MathVector vi = i.getVelocity();
        MathVector e;
        if (vi.length() < 1e-8) {
            e = i.directionToTarget().normalize();
        } else {
            e = vi.normalize();
        }
        return e.dot(rij) >= 0.0;
    }

    public static boolean overlap(Particle i, Particle j, double L) {
        MathVector rij = directDelta(i.getPosition(), j.getPosition());
        double dij = rij.length();
        return dij < (i.getR() + j.getR());
    }

    private static double pointLineDistance(MathVector P, MathVector eHat, MathVector C) {
        double pcx = C.x() - P.x();
        double pcy = C.y() - P.y();
        double cross = pcx * eHat.y() - pcy * eHat.x();
        return Math.abs(cross);
    }

    //@TODO: revisar
    public static boolean frontBandIntersects(Particle i, Particle j, double L) {
        MathVector vi = i.getVelocity();
        MathVector e;
        if (vi.length() < 1e-8) {
            MathVector toT = i.directionToTarget();
            double nt = toT.length();
            if (nt < 1e-8) return false;
            e = toT.scale(1.0/nt);
        } else {
            double nv = vi.length();
            e = vi.scale(1.0/nv);
        }

        MathVector ri = i.getPosition();
        MathVector rj = j.getPosition();

        MathVector nPerp = e.prep().normalize(); // unitario ⟂ a e

        MathVector Pplus  = ri.add(nPerp.scale(i.getRMin()));
        MathVector Pminus = ri.add(nPerp.scale(-i.getRMin()));
        MathVector rjPlus  = Pplus.add(directDelta(Pplus, rj));
        MathVector rjMinus = Pminus.add(directDelta(Pminus, rj));

        double dPlus  = pointLineDistance(Pplus,  e, rjPlus);
        double dMinus = pointLineDistance(Pminus, e, rjMinus);

        boolean hitsBand = (dPlus <= j.getR() + 1e-12) || (dMinus <= j.getR() + 1e-12);
        return hitsBand && isAhead(i, j, L);
    }

    public static boolean contactAcp(Particle i, Particle j, double L) {
        if (Math.abs(i.getR() - i.getRMin()) < 1e-12) {
            return overlap(i, j, L);
        } else {
            return frontBandIntersects(i, j, L) && overlap(i, j, L);
        }
    }

    public static MathVector escapeDir(Particle i, Particle j, double L) {
        MathVector dji = directDelta(j.getPosition(), i.getPosition());
        double n = dji.length();
        if (n < EPS) {
            double a = 2*Math.PI*Math.random();
            return new MathVector(Math.cos(a), Math.sin(a));
        }
        return dji.scale(1.0 / n);
    }

    public static boolean contactWithCircle(Particle i, MathVector r0, double R0, double L) {
        MathVector ri = i.getPosition();
        MathVector dri = directDelta(ri, r0);
        return dri.length() <= (R0 + i.getRMin());
    }
}
