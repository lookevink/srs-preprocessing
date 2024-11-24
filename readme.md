Simple data preprocessing pipeline for SRS data:

1. Start by making a venv to install the requirements into a separate environment: `python3.12 -m venv venv`
2. Activate the venv: `source venv/bin/activate`
3. Install the requirements: `pip3 install -r requirements.txt`
4. Load the OIR files into data/input/
5. Run the pipeline: `python3 -m src.main`

The output will be saved in `data/output/`. You can compare the unstable and stable outputs to see the difference.

Don't hesitate to reach out if you adjustments are needed. We can expand the scope of this to speed & scale up SRS-tailored spectral matching algorithm in general to increase the speed of the detection pipeline.
