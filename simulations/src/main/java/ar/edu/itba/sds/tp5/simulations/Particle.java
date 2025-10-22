package ar.edu.itba.sds.tp5.simulations;

import java.util.Random;

public class Particle {
    public static final double R_MIN_DEFAULT = 0.1;
    public static final double R_MAX_DEFAULT = 0.21;
    public static final double V_DEFAULT = 1.7;
    private static final Random RANDOM = new Random();

    private final double rMin;
    private final double rMax;
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
        this.r = rMin + (rMax - rMin) * RANDOM.nextDouble();
        final double angle = RANDOM.nextDouble() * 2 * Math.PI;
        this.vx = v * Math.cos(angle);
        this.vy = v * Math.sin(angle);
        this.x = RANDOM.nextDouble() * L;
        this.y = RANDOM.nextDouble() * L;
        this.xDesired = RANDOM.nextDouble() * L;
        this.yDesired = RANDOM.nextDouble() * L;
    }

    public Particle(double L) {
        this(R_MIN_DEFAULT, R_MAX_DEFAULT, V_DEFAULT, L);
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
}
