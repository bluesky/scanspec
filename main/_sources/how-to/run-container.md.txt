# Run in a container

Pre-built containers with scanspec and its dependencies already
installed are available on [Github Container Registry](https://ghcr.io/bluesky/scanspec).

## Starting the container

To pull the container from github container registry and run:

```
$ docker run ghcr.io/bluesky/scanspec:latest --version
```

To get a released version, use a numbered release instead of `latest`.
