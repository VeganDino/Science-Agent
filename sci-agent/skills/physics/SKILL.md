---
name: physics
description: "Handle physics tasks: acoustics, mechanics, thermodynamics, electromagnetism, optics, fluid dynamics, quantum mechanics, plasma physics, structural mechanics using SciAgentGYM toolkit functions."
license: MIT
metadata:
  author: science-agent
  version: "1.0"
allowed-tools: run_gym_tool
---

# Physics Skill

Use this skill for physics calculations. All functions are called via:
`run_gym_tool(tool_name, '{"param": value, ...}')`

Use `gym_search_tools(keyword)` to discover specific function names.

## Acoustics
- Sound pressure level calculations, multi-source SPL combination
- Doppler effect: `calculate_doppler_shift`, `doppler_blood_velocity`
- Acoustic spectrum analysis

## Mechanics
- Circular motion, pendulum dynamics, pulley systems
- Friction, conical motion, vibration analysis
- Lagrangian/Hamiltonian mechanics
- Relativistic mechanics

## Thermodynamics
- Heat transfer, entropy, Carnot efficiency
- Phase transitions, thermodynamic cycles
- Ideal gas laws, van der Waals equations

## Electromagnetism
- Electric field, magnetic field calculations
- Circuit analysis, capacitors, inductors
- Maxwell's equations applications

## Optics
- Lens equations, diffraction, interference
- Thin film optics, polarization
- Snell's law, optical path calculations

## Fluid Dynamics
- Bernoulli's equation, viscosity
- Reynolds number, turbulent/laminar flow
- Buoyancy, pressure calculations

## Quantum Mechanics
- Wave functions, probability densities
- Hydrogen atom: `hydrogen_wavefunction_radial`
- Energy levels, quantum numbers

## Structural & Solid Mechanics
- Stress, strain, Young's modulus
- Beam bending, torsion

## Workflow
1. `gym_search_tools(keyword)` — find the exact function name
2. `run_gym_tool(function_name, '{"param1": v1, "param2": v2}')` — call it
3. For complex multi-step problems, spawn a 'physicist' agent via `spawn_agent`
