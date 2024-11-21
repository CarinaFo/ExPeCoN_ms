# Install necessary packages
if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")
if (!requireNamespace("tools", quietly = TRUE)) install.packages("tools")

# Load renv
library(renv)

# Step 1: Detect dependencies from your R scripts
dependencies <- renv::dependencies(path = "E:/expecon_ms/code/Rscripts")

# Step 2: Get installed versions
installed <- as.data.frame(installed.packages(), stringsAsFactors = FALSE)
required_packages <- unique(dependencies$Package)

# Match detected packages with their installed versions
used_packages <- installed[installed$Package %in% required_packages, c("Package", "Version")]

# Save results to a file
write.csv(used_packages, "Rpackages_expecon_ms_inclversions.csv", row.names = FALSE)

# Print to console
print(used_packages)
