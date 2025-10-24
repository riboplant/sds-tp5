package ar.edu.itba.sds.tp5.simulations;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Date;

public class Engine {
    public static void main(String[] args) throws IOException {
        final double L = 6.0;

        final String simulationName = System.getProperty("name", "%d".formatted(new Date().getTime()));

        System.out.println("Running simulation: " + simulationName);

        final Path simulationDirPath = Path.of("data", "simulations", simulationName);
        Files.createDirectories(simulationDirPath);

        try (var writer = Files.newBufferedWriter(simulationDirPath.resolve("static.txt"))) {
            writer.write(String.valueOf(L));
            writer.newLine();
        }
    }
}
