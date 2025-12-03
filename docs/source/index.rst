.. |scbiot-logo| image:: _static/scbiot_logo.svg
   :height: 36px
   :alt: scBIOT logo

scBIOT documentation
==================================

**scBIOT** (Single-Cell Biological Insights via Optimal Transport and Omics Transformers) unifies optimal-transport alignment with Transformer encoders for preprocessing and embedding single-cell RNA, ATAC, and multi-omic data. The library is designed for reproducible benchmarking and scalable inference across modalities.

.. only:: html

   .. raw:: html

      <div class="scbiot-search-hero" role="search" aria-label="Search the scBIOT documentation">
        <form class="scbiot-search-form" action="search.html" method="get">
          <span class="scbiot-search-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <circle cx="11" cy="11" r="7"></circle>
              <line x1="16.65" y1="16.65" x2="21.5" y2="21.5"></line>
            </svg>
          </span>
          <input
            class="scbiot-search-input"
            id="scbiot-search-input"
            type="search"
            name="q"
            placeholder="Search"
            aria-label="Search the documentation"
            autocomplete="off"
          />
          <span class="scbiot-search-shortcut" aria-hidden="true">
            <kbd>Ctrl</kbd><span>+</span><kbd>K</kbd>
          </span>
          <input type="hidden" name="check_keywords" value="yes" />
          <input type="hidden" name="area" value="default" />
        </form>
      </div>

Highlights
----------

* Fast optimal transport with GPU.
* A unified `scBIOT` models that can embed RNA, ATAC, or paired multi-omics  modalities and reuse the fitted pipeline for inference on new batches.
* Supports scRNA-seq, scATAC-seq, paired and unpaired multi-omics.
* Built-in preprocessing steps (iterative LSI).
* Support both CPU and GPU.

.. toctree::
   :maxdepth: 1
   :caption: Get started

   installation
   usage

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api
   preprocessing
   ot
