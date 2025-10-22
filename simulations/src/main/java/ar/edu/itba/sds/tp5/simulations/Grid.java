package ar.edu.itba.sds.tp5.simulations;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class Grid implements Iterable<Particle> {
    private final double L;
    private final List<Particle> particles;

    public Grid(int N, double L) {
        this.L = L;
        this.particles = new ArrayList<>(N);
        for (int i = 0; i < N; i++) {
            particles.add(new Particle(L));
        }
    }

    public double getL() {
        return L;
    }

    public List<Particle> getParticles() {
        return particles;
    }

    @Override
    public Iterator<Particle> iterator() {
        return particles.iterator();
    }
}
