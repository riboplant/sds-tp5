package ar.edu.itba.sds.tp5.simulations.models;

import java.util.Random;

public class Particle {
    private static final Random RANDOM = new Random(12345);
    private static final double EPS = 1e-12;
    private static final double EPS_IN  = 0.12;
    private static final double EPS_OUT = 0.20;
    private static final double TARGET_PADDING = 0.50;
    private static final double TAU = 0.5;
    private static final double BETA = 0.9;
    private static int idCounter = 0;

    private final double rMin;
    private final double rMax;;
    private final double vDesiredMax;
    private final boolean isFixed;
    private final int id;
    private double r;
    private double prevR;
    private MathVector velocity;
    private MathVector position;
    private MathVector prevPosition;
    private MathVector target;
    private boolean arrived = false;

    public Particle(double rMin, double rMax, double v, double L) {
        this.rMin = rMin;
        this.rMax = rMax;
        this.id = idCounter++;
        this.r = rMin + (rMax - rMin) * RANDOM.nextDouble();
        this.prevR = r;
        final double angle = RANDOM.nextDouble() * 2 * Math.PI;
        this.velocity = MathVector.ofPolar(v, angle);
        this.vDesiredMax = v;
        this.position = new MathVector(RANDOM.nextDouble() * L, RANDOM.nextDouble() * L);
        this.prevPosition = this.position;
        this.target = new MathVector(RANDOM.nextDouble() * L, RANDOM.nextDouble() * L);
        isFixed = false;
    }

    public Particle(final boolean isFixed, double rDefault, double L) {
        this.isFixed = isFixed;
        this.rMin = rDefault;
        this.rMax = rDefault;
        this.vDesiredMax = 0.0;
        this.id = idCounter++;
        this.prevR = rDefault;
        this.position = new MathVector(L / 2.0, L / 2.0);
        this.r = rDefault;
    }

    //@TODO: borrar, es de prueba
    public Particle(double x, double y, double tx, double ty, double rMin, double rMax, double v) {
        this.rMin = rMin;
        this.rMax = rMax;
        this.id = idCounter++;
        this.r = rMin + (rMax - rMin) * RANDOM.nextDouble();
        this.prevR = r;
        this.vDesiredMax = v;
        this.position = new MathVector(x, y);
        this.prevPosition = this.position;
        this.target = new MathVector(tx, ty);
        this.velocity = initVelocity();
        isFixed = false;
    }

    private MathVector initVelocity() {
        MathVector et = this.directionToTarget();
        if (et != null && et.length() >= EPS) {
            et = et.normalize();
            return et.scale(speedFromRadius());
        } else {
            return new MathVector(0.0, 0.0);
        }
    }

    public void updateRadius(boolean isCollision, double dt) {
        if (isCollision) {
            this.r = this.rMin;
            this.prevR = this.rMin;
        } else {
            this.r = Math.min(this.r + this.rMax*(dt/TAU), this.rMax);
            this.prevR = this.r;
        }
    }

    public void updateVelocity(boolean isFrontalContact, MathVector e) {
        MathVector dir = e;
        if (dir == null || dir.length() < 1e-12) {
            if (this.velocity != null && this.velocity.length() >= 1e-12) {
                dir = this.velocity.normalize();
            } else {
                //  @TODO: ver si no conviene que este fallback sea random
                dir = new MathVector(1.0, 0.0); // fallback determinista
            }
        } else {
            dir = dir.normalize();
        }

        final double speed = isFrontalContact ? this.vDesiredMax : speedFromRadius();
        this.velocity = dir.scale(speed);
    }

    private double speedFromRadius() {
        double num = Math.max(0.0, this.r - this.rMin);
        double den = Math.max(1e-12, this.rMax - this.rMin);
        double x = num / den;
        double s = vDesiredMax * Math.pow(x, BETA);
        if (Double.isNaN(s) || s < 0) return 0.0;
        return Math.min(s, vDesiredMax);
    }

    public void updatePosition(double dt, double L) {
        this.position = this.position.add(this.velocity.scale(dt));

        // Update target if it has been reached
        double d = this.position.subtract(this.target).length();
        if (!arrived && d <= EPS_IN) {
            MathVector nt = null;
            int tries = 0;
            do {
                nt = randomTargetInBox(L);
                tries++;
            } while (this.position.subtract(nt).length() < TARGET_PADDING && tries < 10);

            if (this.position.subtract(nt).length() < TARGET_PADDING) {
                final double x = (this.position.x() < L*0.5)
                        ? (L*0.5 + RANDOM.nextDouble()*(L*0.5))
                        : (RANDOM.nextDouble()*(L*0.5));
                final double y = (this.position.y() < L*0.5)
                        ? (L*0.5 + RANDOM.nextDouble()*(L*0.5))
                        : (RANDOM.nextDouble()*(L*0.5));
                nt = new MathVector(x, y);
            }

            this.target = nt;
            this.arrived = true;

        } else if (arrived && d >= EPS_OUT) {
            this.arrived = false;
        }
    }

    private MathVector randomTargetInBox(final double L) {
        return new MathVector(RANDOM.nextDouble() * L, RANDOM.nextDouble() * L);
    }

    public MathVector directionToTarget() {
        if (target == null) return MathVector.ZERO;
        return target.subtract(position);
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

    public MathVector getVelocity() { return velocity; }

    public MathVector getPosition() { return position; }

    public int getId() {
        return id;
    }

    public boolean isFixed() {
        return isFixed;
    }
}
