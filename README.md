# Distribution Grid Optimization Engine

## Executive Summary
This project implements a hybrid physics-software engine for analyzing low-voltage distribution grids. It was engineered to address specific DSO challenges such as reverse power flows and voltage violations caused by Distributed Energy Resources (DERs).

## Key Features
- [cite_start]**Physics Core:** Forward-Backward Sweep (FBS) solver optimized for high R/X ratio radial feeders.
- [cite_start]**Domain Modeling:** IEC 61970 (CIM) compliant data models using Pydantic for robust type validation.
- **Advanced Controls:** Fuzzy Logic Controller for Volt-VAR optimization in PV inverters.
- [cite_start]**Analytics:** Hosting Capacity Analysis using binary search ($O(\log N)$) for rapid connection assessment.

## Architecture
- **Layer 1: Domain Data (Pydantic):** Enforces physical constraints (e.g., non-negative resistance) at the I/O boundary.
- **Layer 2: Topology (NetworkX):** Validates radiality and determines feeder ordering.
- **Layer 3: Solvers (SciPy/NumPy):** Implements numerical methods for non-linear power flow.
- **Layer 4: Application:** Orchestrates solvers to answer business questions (e.g., "Can I connect this EV charger?").

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

   Run the simulation and validation suite:

Bash

python main.py
