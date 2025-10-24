package ar.edu.itba.sds.tp5.simulations.models;

import java.util.Random;

public class Particle {
    private static final Random RANDOM = new Random();
    private static final double TAU = 0.5;
    private static int idCounter = 0;

    private final double rMin;
    private final double rMax;
    private final double prevR;
    private final boolean isFixed;
    private final int id;
    private double r;
    private MathVector velocity;
    private MathVector position;
    private MathVector prevPosition;
    private MathVector target;

    public Particle(double rMin, double rMax, double v, double L) {
        this.rMin = rMin;
        this.rMax = rMax;
        this.id = idCounter++;
        this.r = rMin + (rMax - rMin) * RANDOM.nextDouble();
        this.prevR = r;
        final double angle = RANDOM.nextDouble() * 2 * Math.PI;
        this.velocity = MathVector.ofPolar(v, angle);
        this.position = new MathVector(RANDOM.nextDouble() * L, RANDOM.nextDouble() * L);
        this.prevPosition = this.position;
        this.target = new MathVector(RANDOM.nextDouble() * L, RANDOM.nextDouble() * L);
        isFixed = false;
    }

    public Particle(final boolean isFixed, double rDefault, double L) {
        this.isFixed = isFixed;
        this.rMin = rDefault;
        this.rMax = rDefault;
        this.id = idCounter++;
        this.prevR = rDefault;
        this.position = new MathVector(L / 2.0, L / 2.0);
        this.r = rDefault;
    }

    public void updatePosition(boolean isCollision, double dt, double prevR) {
        if(isCollision) {
            this.r = this.rMin;
        } else {
            double newR = this.prevR + this.rMax*(dt/TAU);
        }
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
        return velocity.x();
    }

    public double getVy() {
        return velocity.y();
    }

    public double getX() {
        return position.x();
    }

    public double getY() {
        return position.y();
    }

    public int getId() {
        return id;
    }

    public boolean isFixed() {
        return isFixed;
    }
}
