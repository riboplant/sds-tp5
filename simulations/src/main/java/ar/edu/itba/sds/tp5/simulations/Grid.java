//package ar.edu.itba.sds.tp5.simulations;
//
//import ar.edu.itba.sds.tp5.simulations.models.Particle;
//
//import java.util.ArrayList;
//import java.util.Comparator;
//import java.util.HashMap;
//import java.util.HashSet;
//import java.util.Iterator;
//import java.util.List;
//import java.util.Map;
//import java.util.Set;
//
//public class Grid implements Iterable<Particle> {
//    private final double L;
//    private final List<Particle> particles;
//    private final double dt;
//
//    private static final int[][] NEIGHBOR_OFFSETS = {
//        {0, 1},
//        {1, -1},
//        {1, 0},
//        {1, 1}
//    };
//
//    public Grid(int N, double L, double dt) {
//        this.L = L;
//        this.particles = new ArrayList<>(N);
//        this.dt = dt;
//        for (int i = 0; i < N; i++) {
//            particles.add(new Particle(L));
//        }
//        particles.add(new Particle(true, L));
//    }
//
//    public double getL() {
//        return L;
//    }
//
//    public List<Particle> getParticles() {
//        return particles;
//    }
//
//    public double getDt() {
//        return dt;
//    }
//
//    @Override
//    public Iterator<Particle> iterator() {
//        return particles.iterator();
//    }
//
//    public Map<Integer, List<Particle>> cellIndexMethod(double rc) {
//        if (rc <= 0) {
//            throw new IllegalArgumentException("rc must be positive");
//        }
//        if (particles.isEmpty()) {
//            return Map.of();
//        }
//
//        final double maxRadius = particles.stream()
//            .mapToDouble(Particle::getR)
//            .max()
//            .orElse(0.0);
//
//        final double minCellSide = rc + 2 * maxRadius;
//        int cellsPerSide = minCellSide <= 0 ? 1 : (int) Math.floor(L / minCellSide);
//        if (cellsPerSide < 1) {
//            cellsPerSide = 1;
//        }
//        final double cellSide = L / cellsPerSide;
//
//        final Map<Integer, List<Particle>> occupancy = buildOccupancy(cellsPerSide, cellSide);
//        final Map<Integer, Set<Particle>> contactSets = new HashMap<>();
//
//        for (Map.Entry<Integer, List<Particle>> entry : occupancy.entrySet()) {
//            final int cellIndex = entry.getKey();
//            final List<Particle> cellParticles = entry.getValue();
//            final int row = cellIndex / cellsPerSide;
//            final int column = cellIndex % cellsPerSide;
//
//            // Evaluate contacts within the same cell
//            for (int i = 0; i < cellParticles.size(); i++) {
//                final Particle pi = cellParticles.get(i);
//                for (int j = i + 1; j < cellParticles.size(); j++) {
//                    final Particle pj = cellParticles.get(j);
//                    if (areInContact(pi, pj, rc)) {
//                        addContact(contactSets, cellIndex, pi);
//                        addContact(contactSets, cellIndex, pj);
//                    }
//                }
//            }
//
//            // Evaluate contacts with the relevant neighbor cells
//            for (int[] offset : NEIGHBOR_OFFSETS) {
//                final int neighborRow = row + offset[0];
//                final int neighborColumn = column + offset[1];
//                if (!isInsideGrid(neighborRow, neighborColumn, cellsPerSide)) {
//                    continue;
//                }
//                final int neighborIndex = neighborRow * cellsPerSide + neighborColumn;
//                final List<Particle> neighborParticles = occupancy.get(neighborIndex);
//                if (neighborParticles == null || neighborParticles.isEmpty()) {
//                    continue;
//                }
//                for (Particle pi : cellParticles) {
//                    for (Particle pj : neighborParticles) {
//                        if (areInContact(pi, pj, rc)) {
//                            addContact(contactSets, cellIndex, pi);
//                            addContact(contactSets, cellIndex, pj);
//                            addContact(contactSets, neighborIndex, pi);
//                            addContact(contactSets, neighborIndex, pj);
//                        }
//                    }
//                }
//            }
//        }
//
//        final Map<Integer, List<Particle>> contactsByCell = new HashMap<>();
//        final Comparator<Particle> byId = Comparator.comparingInt(Particle::getId);
//        for (Map.Entry<Integer, Set<Particle>> entry : contactSets.entrySet()) {
//            final List<Particle> cellContacts = new ArrayList<>(entry.getValue());
//            cellContacts.sort(byId);
//            contactsByCell.put(entry.getKey(), cellContacts);
//        }
//        return contactsByCell;
//    }
//
//    private Map<Integer, List<Particle>> buildOccupancy(int cellsPerSide, double cellSide) {
//        final Map<Integer, List<Particle>> occupancy = new HashMap<>();
//        for (Particle particle : particles) {
//            final int cellIndex = toCellIndex(particle, cellsPerSide, cellSide);
//            occupancy.computeIfAbsent(cellIndex, k -> new ArrayList<>()).add(particle);
//        }
//        return occupancy;
//    }
//
//    private int toCellIndex(Particle particle, int cellsPerSide, double cellSide) {
//        final int column = clampToGrid((int) (particle.getX() / cellSide), cellsPerSide);
//        final int row = clampToGrid((int) (particle.getY() / cellSide), cellsPerSide);
//        return row * cellsPerSide + column;
//    }
//
//    private int clampToGrid(int coordinate, int cellsPerSide) {
//        if (coordinate < 0) {
//            return 0;
//        }
//        final int maxIndex = cellsPerSide - 1;
//        if (coordinate > maxIndex) {
//            return maxIndex;
//        }
//        return coordinate;
//    }
//
//    private boolean areInContact(Particle p1, Particle p2, double rc) {
//        final double dx = p1.getX() - p2.getX();
//        final double dy = p1.getY() - p2.getY();
//        final double centerDistance = Math.sqrt(dx * dx + dy * dy);
//        final double surfaceDistance = centerDistance - (p1.getR() + p2.getR());
//        return surfaceDistance <= rc;
//    }
//
//    private void addContact(Map<Integer, Set<Particle>> contactSets, int cellIndex, Particle particle) {
//        contactSets.computeIfAbsent(cellIndex, k -> new HashSet<>()).add(particle);
//    }
//
//    private boolean isInsideGrid(int row, int column, int cellsPerSide) {
//        return row >= 0 && row < cellsPerSide && column >= 0 && column < cellsPerSide;
//    }
//}
