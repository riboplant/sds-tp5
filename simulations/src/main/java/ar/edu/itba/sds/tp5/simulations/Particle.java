package ar.edu.itba.sds.tp5.simulations;

import java.util.Random;

public class Particle {
    public static final double R_MIN_DEFAULT = 0.1;
    public static final double R_MAX_DEFAULT = 0.21;
    public static final double R_FIXED_DEFAULT = 0.21;
    public static final double V_DEFAULT = 1.7;
    private static final Random RANDOM = new Random();
    private static int idCounter = 0;

    private final double rMin;
    private final double rMax;
    private final boolean isFixed;
    private final int id;
    private double r;
    private double vx;
    private double vy;
    private double x;
    private double y;
    private double xDesired;
    private double yDesired;

    public Particle(double rMin, double rMax, double v, double L) {
        this.rMin = rMin;
        this.rMax = rMax;
        this.id = idCounter++;
        this.r = rMin + (rMax - rMin) * RANDOM.nextDouble();
        final double angle = RANDOM.nextDouble() * 2 * Math.PI;
        this.vx = v * Math.cos(angle);
        this.vy = v * Math.sin(angle);
        this.x = RANDOM.nextDouble() * L;
        this.y = RANDOM.nextDouble() * L;
        this.xDesired = RANDOM.nextDouble() * L;
        this.yDesired = RANDOM.nextDouble() * L;
        isFixed = false;
    }

    public Particle(double L) {
        this(R_MIN_DEFAULT, R_MAX_DEFAULT, V_DEFAULT, L);
    }

    public Particle(final boolean isFixed, double L) {
        this.isFixed = isFixed;
        this.rMin = R_FIXED_DEFAULT;
        this.rMax = R_FIXED_DEFAULT;
        this.id = idCounter++;
        this.x = L/2.0;
        this.y = L/2.0;
        this.r = R_FIXED_DEFAULT;
    }

    public double getRMin() {
        return rMin;
    }

    public double getRMax() {
        return rMax;
    }

    public double getR() {
        return r;
    }

    public double getVx() {
        return vx;
    }

    public double getVy() {
        return vy;
    }

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }

    public int getId() {
        return id;
    }

    public boolean isFixed() {
        return isFixed;
    }
}
