package ar.edu.itba.sds.tp5.simulations.models;

import java.util.Objects;

/**
 * Immutable 2D vector with common vector operations.
 */
public record MathVector(double x, double y) {

    public static final MathVector ZERO = new MathVector(0.0, 0.0);
    public static final MathVector UNIT_X = new MathVector(1.0, 0.0);
    public static final MathVector UNIT_Y = new MathVector(0.0, 1.0);

    public MathVector {
        if (Double.isNaN(x) || Double.isNaN(y)) {
            throw new IllegalArgumentException("Vector components must be valid numbers");
        }
        if (!Double.isFinite(x) || !Double.isFinite(y)) {
            throw new IllegalArgumentException("Vector components must be finite");
        }
    }

    public static MathVector ofPolar(double magnitude, double angleRadians) {
        if (magnitude < 0) {
            throw new IllegalArgumentException("Magnitude cannot be negative");
        }
        return new MathVector(
            magnitude * Math.cos(angleRadians),
            magnitude * Math.sin(angleRadians)
        );
    }

    public double length() {
        return Math.hypot(x, y);
    }

    public double lengthSquared() {
        return x * x + y * y;
    }

    public MathVector withLength(double newLength) {
        if (newLength < 0) {
            throw new IllegalArgumentException("New length must be non-negative");
        }
        if (this == ZERO || this.lengthSquared() == 0.0) {
            if (newLength == 0) {
                return ZERO;
            }
            throw new IllegalStateException("Cannot set length for zero vector");
        }
        return normalize().scale(newLength);
    }

    public MathVector normalize() {
        double len = length();
        if (len == 0.0) {
            throw new IllegalStateException("Cannot normalize zero vector");
        }
        return scale(1.0 / len);
    }

    public MathVector negate() {
        return new MathVector(-x, -y);
    }

    public MathVector add(MathVector other) {
        Objects.requireNonNull(other, "other vector");
        return new MathVector(x + other.x, y + other.y);
    }

    public MathVector subtract(MathVector other) {
        Objects.requireNonNull(other, "other vector");
        return new MathVector(x - other.x, y - other.y);
    }

    public MathVector scale(double scalar) {
        return new MathVector(x * scalar, y * scalar);
    }

    public MathVector divide(double scalar) {
        if (scalar == 0.0) {
            throw new IllegalArgumentException("Division by zero");
        }
        return new MathVector(x / scalar, y / scalar);
    }

    public double dot(MathVector other) {
        Objects.requireNonNull(other, "other vector");
        return x * other.x + y * other.y;
    }

    /**
     * Returns the scalar equivalent of the 2D "cross product" (z-component of 3D cross product).
     */
    public double cross(MathVector other) {
        Objects.requireNonNull(other, "other vector");
        return x * other.y - y * other.x;
    }

    public double distanceTo(MathVector other) {
        Objects.requireNonNull(other, "other vector");
        return subtract(other).length();
    }

    public double distanceSquaredTo(MathVector other) {
        Objects.requireNonNull(other, "other vector");
        return subtract(other).lengthSquared();
    }

    /**
     * Angle between this vector and the positive X axis, in radians.
     */
    public double angle() {
        return Math.atan2(y, x);
    }

    /**
     * Signed angle to another vector, in radians, in the range [-pi, pi].
     */
    public double signedAngleTo(MathVector other) {
        Objects.requireNonNull(other, "other vector");
        return Math.atan2(cross(other), dot(other));
    }

    /**
     * Absolute angle between this vector and another vector, in radians.
     */
    public double angleBetween(MathVector other) {
        Objects.requireNonNull(other, "other vector");
        double denominator = length() * other.length();
        if (denominator == 0.0) {
            throw new IllegalStateException("Cannot compute angle with zero vector");
        }
        double value = dot(other) / denominator;
        value = Math.max(-1.0, Math.min(1.0, value));
        return Math.acos(value);
    }

    public MathVector projectOnto(MathVector other) {
        Objects.requireNonNull(other, "other vector");
        double denominator = other.lengthSquared();
        if (denominator == 0.0) {
            throw new IllegalStateException("Cannot project onto zero vector");
        }
        double scalar = dot(other) / denominator;
        return other.scale(scalar);
    }

    public MathVector reflectAcross(MathVector normal) {
        Objects.requireNonNull(normal, "normal vector");
        MathVector n = normal.normalize();
        return subtract(n.scale(2.0 * dot(n)));
    }

    public MathVector lerp(MathVector other, double t) {
        Objects.requireNonNull(other, "other vector");
        return new MathVector(
            x + (other.x - x) * t,
            y + (other.y - y) * t
        );
    }

    public MathVector rotate(double angleRadians) {
        double cos = Math.cos(angleRadians);
        double sin = Math.sin(angleRadians);
        return new MathVector(
            x * cos - y * sin,
            x * sin + y * cos
        );
    }

    public MathVector perpendicular() {
        return new MathVector(-y, x);
    }
}
