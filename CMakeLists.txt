# Create a custom library
add_library(
    rl_tcp_gym
    sim.cc
)

# Link the custom library to the scratch executable
build_exec(
    EXECNAME rl_tcp_gym
    SOURCE_FILES
        sim.cc
        tcp-rl-env.cc
        tcp-rl.cc
    LIBRARIES_TO_LINK 
        rl_tcp_gym
        ${libai}
        ${libcore}
        ${libpoint-to-point}
        ${libpoint-to-point-layout}
        ${libnetwork}
        ${libapplications}
        ${libmobility}
        ${libcsma}
        ${libinternet}
        ${libwifi}
        ${libflow-monitor}
  # use "${ns3-libs}" "${ns3-contrib-libs}" in case you want to link to all
  # enabled modules
  EXECUTABLE_DIRECTORY_PATH ${CMAKE_OUTPUT_DIRECTORY}/scratch/reinforcement-learning
)
