Here is an easy to run python tool for visualizing experiments simply by uploading excels with their data!
# Graphite: Experimental Data Visualization Dashboard

An interactive web application for visualizing radiation, hypergravity, and spaceflight experimental data

## Features

### Universal Data Integration
- **Plug-and-play Excel support**: Upload any properly formatted Excel file
- **Automatic parsing**: Dynamically detects payloads, experimental groups, and parameters
- **Flexible structure**: Adapts to your dataset without code changes

### Interactive Visualizations
- **Multi-axis views**: Toggle between Time, Dose, or Animal Count on x-axis
- **Smart hover tooltips**: Customize displayed parameters in real-time
- **Dose-responsive icons**: Radiation symbols scale with intensity
- **Auto-wrapped labels**: Y-axis formatting for complex group names

### Experiment-Specific Modules
- **Ground Radiation**: Dose-ordered groups with NSRL metadata support
- **Hypergravity**: Combined gravity/radiation visualization
- **Spaceflight**: Gravity-level color coding (1G, uG, etc.)
- **Extensible framework**: Easy to add new experiment types

## Sample Data Structure

Your Excel file should include these columns:
- `Payload ID (rdrc_name)`
- `Experimental Group (rdrc_name)` 
- `Time point of sacrifice post irradiation`
- `Total absorbed dose`
- `Gravity level` (for spaceflight/hypergravity)
- Measurement parameters (any numeric columns)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/graphite.git
cd graphite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r 

dash>=2.0.0
plotly>=5.0.0
pandas>=1.3.0
dash-bootstrap-components>=1.0.0
openpyxl>=3.0.0

# Manual - you can also just download the python file from github and run.


# In browser (http://localhost:8050 )
