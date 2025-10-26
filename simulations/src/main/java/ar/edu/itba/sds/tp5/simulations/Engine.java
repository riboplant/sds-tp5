package ar.edu.itba.sds.tp5.simulations;

import ar.edu.itba.sds.tp5.simulations.models.Particle;
import ar.edu.itba.sds.tp5.simulations.models.Pedestrians;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Date;
import java.util.Locale;

public class Engine {
    public static void main(String[] args) throws IOException {
        Locale.setDefault(Locale.US);
        final double L = 6.0;
        final double vDesiredMax = 1.7;
        final double rMin = 0.1;
        final double rMax = 0.21;

        final String simulationName = System.getProperty("name", "%d".formatted(new Date().getTime()));
        //@TODO: cambiar el default
        final int N = Integer.parseInt(System.getProperty("N", "4"));
        final double dt = Double.parseDouble(System.getProperty("dt", "%f".formatted(1.0 / 33.0)));
        final int step = Integer.parseInt(System.getProperty("step", "1"));
        final double t_f = Double.parseDouble(System.getProperty("t_f", "10.0"));

        System.out.printf("%d%n", N);
        System.out.println("Running simulation: " + simulationName);

        final Pedestrians pedestrians = new Pedestrians(N, L, rMin, rMax, vDesiredMax);
        final Path simulationDirPath = Path.of("..","data", "simulations", simulationName);
        Files.createDirectories(simulationDirPath);
        try (var writer = Files.newBufferedWriter(simulationDirPath.resolve("static.txt"))) {
            writer.write(String.valueOf(L));
            writer.newLine();
            writer.write(String.valueOf(N));
            writer.newLine();
            writer.write(String.valueOf(rMin));
            writer.newLine();
            for(final Particle p : pedestrians) {
                writer.write(p.isFixed() ? "F" : "M");
                writer.newLine();
            }
        }

        int t = 0;
        try (var writer = Files.newBufferedWriter(simulationDirPath.resolve("dynamic.txt"))) {
            do {
                if(t % step == 0) {
                    writer.write(String.valueOf(pedestrians.getTime()));
                    writer.newLine();
                    for(final Particle p : pedestrians) {
                        writer.write("%.12f %.12f %.12f %.12f %.12f".formatted(
                                        p.getPosition().x(),
                                        p.getPosition().y(),
                                        p.getVelocity().x(),
                                        p.getVelocity().y(),
                                        p.getR()
                                )
                        );
                        writer.newLine();
                    }
                }
                t++;
                pedestrians.step(dt);
            } while (pedestrians.getTime() < t_f);
        }

        System.out.println("System created with " + N + " particles.");
    }
}
