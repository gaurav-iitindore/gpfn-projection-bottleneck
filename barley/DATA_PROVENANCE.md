# Data Provenance, Barley HEB-25

## Genotypes
- File: C_IBS_imputed_MNI.csv
- Source: Maurer & Pillen 2019, IPK Gatersleben e!DAL repository
- DOI: 10.5447/ipk/2019/20
- Dimensions: 33,005 markers x 1,363 lines
- Processing: fractional imputation dosages (0.26% of cells) hard-called to
  nearest 0/1/2. Documented; matches the SoyNAM hard-call regime.

## Physical map
- File: snp_effects.txt (barley 50k iSelect array)
- Source: Bayer et al. 2017, Front. Plant Sci. 8:1792 (Morex reference)
- URL: https://ics.hutton.ac.uk/50k/resources/snp_effects.txt
- Reconciliation: underscore-form vs hyphen-form marker IDs matched via sed
  conversions; 32,833 of 33,005 markers mapped across 7 chromosomes
  (172 dropped as unmappable/blank).

## Phenotypes
- TGW (thousand-grain weight, polygenic): 8 environments
  (Halle/Dundee x 2014/2015 x N0/N1), from Sharma et al. 2017.
- FT (flowering-time BLUEs, oligogenic): Maurer et al. 2015, BMC Genomics 16:290,
  Additional file 5, column FT_BLUEs; 1,363 lines with genotype + phenotype.

## Model
- pika.pt (NAM/structured GPFN, Ubbens et al. 2025). mongoose.pt not used.

## Reference genome
- Morex (barley reference), consistent with the 50k array design.
