# Run in a container

Pre-built containers with scanspec and its dependencies already
installed are available on [Github Container Registry](https://ghcr.io/dls-controls/scanspec).

## Starting the container

To pull the container from github container registry and run:

```
$ docker run ghcr.io/dls-controls/scanspec:main --version
```

To get a released version, use a numbered release instead of `main`.
