import unittest
from unittest.mock import patch
import pandas as pd
import matplotlib.pyplot as plt
from utilis import plot_building_kpis, plot_district_kpis  # Assuming the functions are in utilis.py
import os

# Define the directory to save the plots
save_dir = '/data/mengxin/ping/CityLearn/results/2023/Plot_test_new_metrics'

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

class TestPlotKPIs(unittest.TestCase):

    @patch('utilis.get_kpis')
    def test_plot_building_kpis_mixed(self, mock_get_kpis):
        """Test plot_building_kpis when some KPIs are zero and some are non-zero."""

        # Mock data: Some KPIs have zero values, others do not
        mock_get_kpis.return_value = pd.DataFrame({
            'name': ['building_1', 'building_1', 'building_2', 'building_2', 'building_3', 'building_3'],
            'kpi': ['Cost', 'Emissions', 'Cost', 'Emissions', 'Cost', 'Emissions'],
            'value': [0.9, 0.0, 0.78, 0.0, 1.01, 0.5],
            'level': ['building', 'building', 'building', 'building', 'building', 'building']
        })

        # Mock environment dictionary
        envs = {'agent_1': 'env_1', 'agent_2': 'env_2'}

        # Call the function and ensure that it runs
        fig = plot_building_kpis(envs)
        self.assertIsInstance(fig, plt.Figure)  # Ensure it returns a figure

        # Save the figure
        fig.savefig(os.path.join(save_dir, 'building_level_kpis_mixed.png'), dpi=150)

    @patch('utilis.get_kpis')
    def test_plot_building_kpis_all_zero(self, mock_get_kpis):
        """Test plot_building_kpis when all KPIs are zero, should skip plotting."""

        # Mock data: All KPIs have zero values
        mock_get_kpis.return_value = pd.DataFrame({
            'name': ['building_1', 'building_1', 'building_2', 'building_2', 'building_3', 'building_3'],
            'kpi': ['Cost', 'Emissions', 'Cost', 'Emissions', 'Cost', 'Emissions'],
            'value': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'level': ['building', 'building', 'building', 'building', 'building', 'building']
        })

        # Mock environment dictionary
        envs = {'agent_1': 'env_1', 'agent_2': 'env_2'}

        # Call the function and ensure that no plot is created
        fig = plot_building_kpis(envs)
        self.assertIsNone(fig)  # Check that no figure is returned if all KPIs are zero

    @patch('utilis.get_kpis')
    def test_plot_building_kpis_all_non_zero(self, mock_get_kpis):
        """Test plot_building_kpis when all KPIs have non-zero values."""

        # Mock data: All KPIs have non-zero values
        mock_get_kpis.return_value = pd.DataFrame({
            'name': ['building_1', 'building_1', 'building_2', 'building_2', 'building_3', 'building_3'],
            'kpi': ['Cost', 'Emissions', 'Cost', 'Emissions', 'Cost', 'Emissions'],
            'value': [0.9, 0.5, 0.78, 0.6, 1.01, 0.5],
            'level': ['building', 'building', 'building', 'building', 'building', 'building']
        })

        # Mock environment dictionary
        envs = {'agent_1': 'env_1', 'agent_2': 'env_2'}

        # Call the function and ensure that it runs
        fig = plot_building_kpis(envs)
        self.assertIsInstance(fig, plt.Figure)  # Ensure it returns a figure

        # Save the figure
        fig.savefig(os.path.join(save_dir, 'building_level_kpis_non_zero.png'), dpi=150)

    @patch('utilis.get_kpis')
    def test_plot_district_kpis_mixed(self, mock_get_kpis):
        """Test plot_district_kpis when some KPIs are zero and some are non-zero."""

        # Mock data: Some KPIs have zero values, others do not
        mock_get_kpis.return_value = pd.DataFrame({
            'name': ['district_1', 'district_1', 'district_2', 'district_2'],
            'kpi': ['Cost', 'Emissions', 'Cost', 'Emissions'],
            'value': [0.9, 0.0, 0.78, 0.0],
            'level': ['district', 'district', 'district', 'district']
        })

        # Mock environment dictionary
        envs = {'agent_1': 'env_1', 'agent_2': 'env_2'}

        # Call the function and ensure that it runs
        fig = plot_district_kpis(envs)
        self.assertIsInstance(fig, plt.Figure)  # Ensure it returns a figure

        # Save the figure
        fig.savefig(os.path.join(save_dir, 'district_kpis_mixed.png'), dpi=150)

    @patch('utilis.get_kpis')
    def test_plot_district_kpis_all_zero(self, mock_get_kpis):
        """Test plot_district_kpis when all KPIs are zero, should skip plotting."""

        # Mock data: All KPIs have zero values
        mock_get_kpis.return_value = pd.DataFrame({
            'name': ['district_1', 'district_1', 'district_2', 'district_2'],
            'kpi': ['Cost', 'Emissions', 'Cost', 'Emissions'],
            'value': [0.0, 0.0, 0.0, 0.0],
            'level': ['district', 'district', 'district', 'district']
        })

        # Mock environment dictionary
        envs = {'agent_1': 'env_1', 'agent_2': 'env_2'}

        # Call the function and ensure that no plot is created
        fig = plot_district_kpis(envs)
        self.assertIsNone(fig)  # Check that no figure is returned if all KPIs are zero

    @patch('utilis.get_kpis')
    def test_plot_district_kpis_all_non_zero(self, mock_get_kpis):
        """Test plot_district_kpis when all KPIs have non-zero values."""

        # Mock data: All KPIs have non-zero values
        mock_get_kpis.return_value = pd.DataFrame({
            'name': ['district_1', 'district_1', 'district_2', 'district_2'],
            'kpi': ['Cost', 'Emissions', 'Cost', 'Emissions'],
            'value': [0.9, 0.5, 0.78, 0.6],
            'level': ['district', 'district', 'district', 'district']
        })

        # Mock environment dictionary
        envs = {'agent_1': 'env_1', 'agent_2': 'env_2'}

        # Call the function and ensure that it runs
        fig = plot_district_kpis(envs)
        self.assertIsInstance(fig, plt.Figure)  # Ensure it returns a figure

        # Save the figure
        fig.savefig(os.path.join(save_dir, 'district_kpis_non_zero.png'), dpi=150)


if __name__ == '__main__':
    unittest.main(exit=False)
