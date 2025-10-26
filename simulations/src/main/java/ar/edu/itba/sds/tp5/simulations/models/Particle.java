package ar.edu.itba.sds.tp5.simulations.models;

import java.util.ArrayList;
import java.util.List;
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
    private final double L;
    private static final List<MathVector> FIXED_CENTERS = new ArrayList<>();
    private static final List<Double> FIXED_RADII = new ArrayList<>();
    private static final double FIXED_TARGET_BUFFER = 0.05;
    private static final int MAX_TARGET_SAMPLING_ATTEMPTS = 1_000;
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
        this.L = L;
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
        this.velocity = MathVector.ZERO;
        this.id = idCounter++;
        this.prevR = rDefault;
        this.position = new MathVector(L / 2.0, L / 2.0);
        this.r = rDefault;
        this.L = L;
        registerFixedObstacle(this.position, this.r);
    }

    //@TODO: borrar, es de prueba
    public Particle(double x, double y, double tx, double ty, double rMin, double rMax, double v, double L) {
        this.rMin = rMin;
        this.rMax = rMax;
        this.id = idCounter++;
        this.r = rMin + (rMax - rMin) * RANDOM.nextDouble();
        this.prevR = r;
        this.vDesiredMax = v;
        this.position = new MathVector(x, y);
        this.prevPosition = this.position;
        this.target = new MathVector(tx, ty);
        this.L = L;
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
        if(isFixed) return;
        if (isCollision) {
            this.r = this.rMin;
            this.prevR = this.rMin;
        } else {
            this.r = Math.min(this.r + this.rMax*(dt/TAU), this.rMax);
            this.prevR = this.r;
        }
    }

    public void updateVelocity(boolean isFrontalContact, MathVector e) {
        if(isFixed) return;
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
        if(isFixed) return;
        this.position = wrapPosition(this.position.add(this.velocity.scale(dt)));

        // Update target if it has been reached
        double d = MathVector.minImage(this.position, this.target, this.L).length();
        if (!arrived && d <= EPS_IN) {
            MathVector nt = null;
            int tries = 0;
            do {
                nt = randomTargetInBox(this.L);
                tries++;
            } while ((MathVector.minImage(this.position, nt, this.L).length() < TARGET_PADDING) && tries < 10);

            if (MathVector.minImage(this.position, nt, this.L).length() < TARGET_PADDING) {
                final double halfL = this.L * 0.5;
                final double x = (this.position.x() < halfL)
                        ? (halfL + RANDOM.nextDouble() * halfL)
                        : (RANDOM.nextDouble() * halfL);
                final double y = (this.position.y() < halfL)
                        ? (halfL + RANDOM.nextDouble() * halfL)
                        : (RANDOM.nextDouble() * halfL);
                nt = new MathVector(x, y);
                if (isInsideFixedObstacle(nt)) {
                    nt = pushOutsideFixedObstacle(nt);
                }
            }

            this.target = nt;
            this.arrived = true;

        } else if (arrived && d >= EPS_OUT) {
            this.arrived = false;
        }
    }

    public MathVector directionToTarget() {
        if (target == null) return MathVector.ZERO;
        MathVector delta = MathVector.minImage(position, target, this.L);
        if (delta.length() < EPS) {
            return MathVector.ZERO;
        }
        return delta;
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

    private MathVector wrapPosition(MathVector pos) {
        return pos.wrapToBox(this.L);
    }

    private static void registerFixedObstacle(MathVector center, double radius) {
        FIXED_CENTERS.add(center);
        FIXED_RADII.add(radius);
    }

    public static void resetStatics() {
        idCounter = 0;
        FIXED_CENTERS.clear();
        FIXED_RADII.clear();
    }

    private MathVector randomTargetInBox(final double L) {
        MathVector candidate = null;
        for (int attempt = 0; attempt < MAX_TARGET_SAMPLING_ATTEMPTS; attempt++) {
            candidate = uniformPointInBox(L);
            if (!isInsideFixedObstacle(candidate)) {
                return candidate;
            }
        }
        return (candidate != null && !isInsideFixedObstacle(candidate))
                ? candidate
                : pushOutsideFixedObstacle(uniformPointInBox(L));
    }

    private MathVector uniformPointInBox(final double L) {
        return new MathVector(RANDOM.nextDouble() * L, RANDOM.nextDouble() * L);
    }

    private boolean isInsideFixedObstacle(MathVector point) {
        if (FIXED_CENTERS.isEmpty()) {
            return false;
        }
        for (int i = 0; i < FIXED_CENTERS.size(); i++) {
            MathVector center = FIXED_CENTERS.get(i);
            double radius = FIXED_RADII.get(i) + FIXED_TARGET_BUFFER;
            double distance = MathVector.minImage(point, center, this.L).length();
            if (distance < radius) {
                return true;
            }
        }
        return false;
    }

    private MathVector pushOutsideFixedObstacle(MathVector point) {
        MathVector adjusted = point;
        for (int i = 0; i < FIXED_CENTERS.size(); i++) {
            MathVector center = FIXED_CENTERS.get(i);
            double radius = FIXED_RADII.get(i) + FIXED_TARGET_BUFFER;
            MathVector delta = MathVector.minImage(center, adjusted, this.L);
            double distance = delta.length();
            if (distance < radius) {
                MathVector direction;
                if (distance < EPS) {
                    double angle = RANDOM.nextDouble() * 2.0 * Math.PI;
                    direction = MathVector.ofPolar(1.0, angle);
                } else {
                    direction = delta.scale(1.0 / distance);
                }
                double push = radius - distance + 1e-3;
                adjusted = wrapPosition(adjusted.add(direction.scale(push)));
            }
        }
        return adjusted;
    }
}
