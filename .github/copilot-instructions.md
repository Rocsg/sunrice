#Overarching goal
This project is about developing 3D rendering and characterisation tools to explore rice root anatomy, from 3D xray observations. We target to document and visualize the compromise between water and gas conduction, and make them explorable, quantifiable, playable, in flatscreen and in VR.


#Current goals
We are currently working on marvel-water-conductance scripts ecosystem.
marvel-water-conductance-build-meshes is the script that prebuilds the meshes and vtk objects
marvel-wind-field-build builds a specific part about gas conduction rendering, involving heavy computation
marvel-water-conductance is the actual rendering script, with interactors and view
marvel-water-movie leverages these to compute a movie, in flatscreen or VR


#Operations localization
I work on my laptop for developing, rendering, and small tests, with this local configuration:
User: "rfernandez"
Code path: "/home/rfernandez/Dev/Python/sunrice"
Venv path: "/home/rfernandez/Dev/venvs/vtk_venv"
Data path: "/home/rfernandez/Data/Arize/Hollow_test"

And I work as well on phenodrone, that is a server I access on ssh, for heavy computation, and that will maybe do rendering one day when things go too heavy for my laptop
User: "romain"
Code path: "/mnt/e823c70f-4136-47c9-91be-1ca7901a37b5/romain_temp_data/Test_Hollow/Dev/sunrice"
Venv path: "/mnt/e823c70f-4136-47c9-91be-1ca7901a37b5/romain_temp_data/Test_Hollow/Dev/vtk_venv/"
Data path: "/mnt/e823c70f-4136-47c9-91be-1ca7901a37b5/romain_temp_data/Test_Hollow/Data"

# Code graph / MCP usage
When needed, use the available code-review-graph / MCP tools to understand the repository structure.
Identify:
- the minimal relevant files,
- the main entry points,
- callers and callees of the modified functions,
- impacted tests or scripts,
- dependency radius.
Do not scan the whole repository unless the graph is missing or insufficient.

#Continuous improvement
If during a chat session, you identify that that would have been nice I add something in these instructions to improve efficience, speed, or limit token usage, you can tell me.

# RTK — Token-Optimized CLI

Use `rtk` for noisy terminal commands whose output may consume many tokens.

Prefer:
- `rtk git status`
- `rtk git diff`
- `rtk git log -10`
- `rtk pytest`
- `rtk grep`
- `rtk find`
- `rtk docker ps`
- `rtk docker logs`

Do not use `rtk` for:
- `cd`, `pwd`, `source`, `which`, `echo`
- venv activation
- short commands with tiny output
- commands where exact raw output is needed

If exact output matters, run the command normally or use `rtk proxy <cmd>`.


