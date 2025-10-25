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

    // @TODO chequear
    public static boolean isAhead(Particle i, Particle j, double L) {
        MathVector rij = minImage(i.getPosition(), j.getPosition(), L);                 // i -> j
        MathVector vi = i.getVelocity();                  // velocidad actual de i
        if (vi.length() < 1e-8) return false;
        MathVector e = vi.normalize();
        return e.dot(rij) >= 0.0;
    }

    public static boolean overlap(Particle i, Particle j, double L) {
        MathVector rij = minImage(i.getPosition(), j.getPosition(), L);
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
        double nv = vi.length();
        if (nv < 1e-8) return false;

        MathVector e = new MathVector(vi.x() / nv, vi.y() / nv);

        MathVector ri = i.getPosition();
        MathVector rj = j.getPosition();

        MathVector nPerp = e.prep().normalize(); // unitario ⟂ a e

        MathVector Pplus  = ri.add(nPerp.scale(i.getRMin()));
        MathVector Pminus = ri.add(nPerp.scale(-i.getRMin()));
        MathVector rjPlus  = Pplus.add(minImage(Pplus, rj, L));
        MathVector rjMinus = Pminus.add(minImage(Pminus, rj, L));

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
        MathVector dji = minImage(j.getPosition(), i.getPosition(), L);
        double n = dji.lengthSquared();
        if (n < EPS) return new MathVector(1, 0);
        return new MathVector(dji.x() / n, dji.y() / n);
    }

    public static boolean contactWithCircle(Particle i, MathVector r0, double R0, double L) {
        MathVector ri = i.getPosition();
        MathVector dri = minImage(ri, r0, L);
        return dri.length() <= (R0 + i.getRMin());
    }
}
