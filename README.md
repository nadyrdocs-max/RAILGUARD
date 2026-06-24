# RAILGUARD
RailGuard: Vibration-based predictive maintenance for railway wheel flats. Bridges classical mechanics (2-DOF Hertzian contact simulation) with Edge AI. Includes scripts for dataset generation, FFT/CWT analysis, and ML severity classification. Built for INFOMATRIX 2027.
 RailGuard Research Simulation Framework
 
Project Overview

RailGuard is an AI-assisted railway wheel defect detection system focused on identifying wheel-flat defects through vibration analysis.

This repository contains the physics-based simulation environment used during the early research stage of the project before hardware validation.

The purpose of the simulation is not to reproduce full-scale railway dynamics exactly, but to generate realistic wheel-flat impact signatures and investigate whether vibration features can reliably distinguish different defect severities.

Research Objective

The main research question is:

Can wheel-flat defects be detected and quantified using vibration signals collected from axle-box mounted accelerometers?

The simulation investigates the relationship between:

Wheel-flat size
Rotational speed
Vibration response
Statistical vibration features

such as:

Kurtosis
Impulse Factor
Crest Factor
Spectral Energy
Project Scope

The simulation is intended to reproduce:

Periodic wheel-flat impacts
Transient vibration events
Relative defect severity trends
Feature extraction workflow
Dataset generation for machine learning experiments

The simulation is not intended to reproduce:

Full railway vehicle dynamics
Real contact stresses
Exact axle loads
Absolute railway vibration amplitudes

Therefore, the simulation serves as a:

Qualitative validation platform rather than a full-scale railway model.

Physical Model

The simulation is based on a simplified dynamic model consisting of:

Wheel

Parameters:

Wheel radius
Rotational speed
Flat defect geometry
Contact Mechanics

Wheel-flat impacts are generated when the flat region contacts the rolling surface.

The impact excitation is modeled as a transient force pulse whose magnitude depends on:

Flat length
Wheel radius
Contact stiffness
Suspension Dynamics

The vibration response is represented using a multi-degree-of-freedom mass-spring-damper system.

Components:

Wheel mass
Bogie mass
Suspension stiffness
Suspension damping

This allows simulation of:

Resonance effects
Impact transmission
Vibration attenuation
Simulation Parameters

Typical simulation ranges:

Parameter	Values
Flat size	0–45 mm
Speed	5–20 m/s
Sampling rate	10 kHz
Duration	1–2 s
Noise level	Randomized
Monte Carlo Validation

To evaluate robustness, multiple realizations are generated for every operating condition.

For each:

Flat size
Speed

multiple repetitions are simulated using different noise realizations.

This produces statistical distributions instead of single measurements.

Benefits:

Robustness assessment
Repeatability evaluation
Feature stability analysis
Generated Dataset

Dataset file:

railguard_robust_dataset.csv

Contains:

Column	Description
FlatSize	Wheel-flat size (mm)
Speed	Wheel speed (m/s)
Repeat	Monte Carlo iteration
RMS	Root Mean Square
Peak	Maximum amplitude
CrestFactor	Peak/RMS ratio
Kurtosis	Impulsiveness indicator
Skewness	Signal asymmetry
ShapeFactor	Waveform shape descriptor
ImpulseFactor	Impact severity metric
SpectralEntropy	Frequency complexity metric
BandEnergy_1000_3000Hz	High-frequency impact energy
Signal Processing Workflow

Raw acceleration signal

↓

Time-domain analysis

↓

FFT spectrum

↓

STFT spectrogram

↓

Feature extraction

↓

Dataset generation

↓

Machine learning preparation

Generated Figures
Time Domain Signals

Shows acceleration response over time.

Used to visualize:

Impact occurrence
Signal amplitude
Defect signatures

Files:

healthy_time_domain.png
defective_time_domain.png
FFT Power Spectrum

Frequency-domain representation.

Used to identify:

Dominant frequencies
Harmonics
Resonance regions

Files:

healthy_fft.png
defective_fft.png
STFT Spectrogram

Time-frequency representation.

Used to visualize:

Impact timing
Frequency evolution
Transient events

Files:

healthy_stft.png
defective_stft.png
Kurtosis Validation

Shows how kurtosis changes with defect severity.

Expected behavior:

Higher defect severity
More impulsive vibration
Increased kurtosis

File:

kurtosis_vs_flat_size.png
Impulse Factor Validation

Measures transient impact strength.

Expected behavior:

Larger flats
Larger impacts
Higher impulse factor

File:

impulse_factor_vs_flat_size.png
Interpretation of Results

The simulation demonstrates that:

Wheel-flat defects generate periodic impacts.
Defect severity influences vibration statistics.
Time-domain features respond to increasing damage.
Frequency-domain energy distribution changes with defect size.

These observations support the feasibility of vibration-based defect detection.

Limitations

The current simulation has several limitations.

Scale Effects

The wheel diameter is significantly smaller than a real railway wheel.

Therefore:

Absolute vibration amplitudes are not representative.
Absolute forces are not representative.

Only relative trends are considered meaningful.

Simplified Contact

Real wheel-rail interaction includes:

Elastic deformation
Creepage
Rail flexibility
Nonlinear contact dynamics

These effects are simplified.

Noise Model

Current simulations use synthetic random noise.

Future hardware experiments will replace synthetic noise with measured sensor data.

Hardware Validation Phase

The next phase of RailGuard involves construction of a laboratory test rig.

Planned hardware:

Raspberry Pi 5
ADXL355 accelerometer
Hall-effect encoder
Load cell
DC motor
Interchangeable wheels
Steel roller

The purpose of the hardware stage is to validate whether trends observed in simulation are reproduced experimentally.

Future Work

Future development includes:

Experimental data collection
Model calibration
Real-world vibration measurements
Machine learning classifier training
Domain adaptation between simulation and hardware data
Real-time defect detection system
Citation

If using this work in reports or presentations, cite as:

RailGuard: Physics-Based Simulation Framework for Wheel-Flat Detection Using Vibration Analysis, 2026.

Author

Nadyr
RailGuard Research Project
AI-Based Railway Wheel Defect Detection System
Target Competition: Infomatrix 2027
