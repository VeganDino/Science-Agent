---
name: astronomy
description: "Handle astronomy tasks: stellar physics, orbital mechanics, cosmology, spectroscopy, telescope calculations using SciAgentGYM astronomy toolkit functions."
license: MIT
metadata:
  author: science-agent
  version: "1.0"
allowed-tools: run_gym_tool
---

# Astronomy Skill

Use this skill for astronomy and astrophysics calculations.
All functions called via: `run_gym_tool(tool_name, '{"param": value}')`

Use `gym_search_tools(keyword)` with terms like:
- 'stellar', 'orbit', 'luminosity', 'redshift', 'telescope', 'magnitude'

## Key Areas (51 functions available)
- Stellar physics: luminosity, temperature, radius, mass
- Orbital mechanics: Kepler's laws, escape velocity, orbital period
- Cosmology: Hubble constant, redshift, distance modulus
- Spectroscopy: spectral lines, Doppler redshift
- Telescope: angular resolution, magnification, light gathering

## Workflow
1. `gym_search_tools('astronomy keyword')` — find function
2. `run_gym_tool(function_name, '{"param": value}')` — execute
