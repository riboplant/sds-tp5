package ar.edu.itba.sds.tp5.simulations.models;

public final class Contact {

    private static final double EPS = 1e-12;
    private Contact(){}

//    public static MathVector minImage(MathVector from, MathVector to, double L) {
//        double dx = to.x() - from.x();
//        double dy = to.y() - from.y();
//        dx -= L * Math.round(dx / L);
//        dy -= L * Math.round(dy / L);
//        return new MathVector(dx, dy);
//    }

    public static MathVector directDelta(MathVector from, MathVector to, double L) {
        return MathVector.minImage(from, to, L);
    }

    // @TODO chequear
    public static boolean isAhead(Particle i, Particle j, double L) {
        MathVector rij = directDelta(i.getPosition(), j.getPosition(), L);
        MathVector vi = i.getVelocity();
        MathVector e;
        if (vi.length() < 1e-8) {
            e = i.getDirectionToTarget();
        } else {
            e = vi.normalize();
        }
        return e.dot(rij) >= 0.0;
    }

    public static boolean overlap(Particle i, Particle j, double L) {
        MathVector rij = directDelta(i.getPosition(), j.getPosition(), L);
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
            MathVector toT = i.getDirectionToTarget();
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
        MathVector rjPlus  = Pplus.add(directDelta(Pplus, rj, L));
        MathVector rjMinus = Pminus.add(directDelta(Pminus, rj, L));

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
        MathVector eij = MathVector.minImage(j.getPosition(), i.getPosition(), L);
        double n = eij.length();
        if (n >= EPS) {
            return eij.scale(1.0 / n);
        }
        MathVector vi = (i.getVelocity() != null) ? i.getVelocity() : MathVector.ZERO;
        MathVector vj = (j.getVelocity() != null) ? j.getVelocity() : MathVector.ZERO;
        MathVector vij = vj.subtract(vi);
        if (vij.length() >= EPS) {
            return vij.scale(-1.0 / vij.length());
        }
        MathVector et = i.getDirectionToTarget();
        if (et != null && et.length() >= EPS) return et.normalize();
        return new MathVector(1.0, 0.0);
    }
}
