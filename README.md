<p align="center">
  <img src="https://raw.githubusercontent.com/tsilva/parselabs/main/logo.png" alt="parselabs" width="512" />
  <br />
  <!-- repo-tagline:start -->
  <strong>🔬 Extract structured lab results from medical PDFs 📊</strong>
  <!-- repo-tagline:end -->
</p>

parselabs extracts objective lab results from report PDFs and images into
structured data for review. It preserves printed values, units, and comparison
signs. Use the review workflow for corrections and interpretation.

## Get started

Install Python 3.11+, uv, and Poppler. Copy the synthetic profile template
to your private config, then follow the [installation and profile steps](https://github.com/tsilva/parselabs/blob/main/docs/parselabs-guide.md#install).
Run the parser with your configured model gateway and inspect the extracted
results before incorporating them into a health record.

See the [extraction pipeline](https://github.com/tsilva/parselabs/blob/main/docs/pipeline.md) for processing and validation.
Keep source documents and live profiles outside the repository.

[Parselabs setup, review, and maintenance](https://github.com/tsilva/parselabs/blob/main/docs/parselabs-guide.md).

## License

[MIT License](https://github.com/tsilva/parselabs/blob/main/LICENSE).
