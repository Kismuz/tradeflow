****

Instruments for easy algorithmic trading tasks modelling with
dataflow graph approach.

****

Makes heavy use of:

- `Pythonflow` library from Spotify guys:
https://github.com/spotify/pythonflow

- `Ray` and `RlLib` libraries: https://github.com/ray-project/ray

****

Main abstractions:
````
    Environment: {
                    API methods: {
                                    reset,
                                    step,
                                  },
                    Graph: {
                            Operations/Nodes: {
                                                Kernels
                                              },
                            States/edges
                           }
                  }

                

```

