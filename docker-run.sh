#!/usr/bin/env bash
docker run \
    --volume /etc/passwd:/etc/passwd:ro \
    --user "$(id -u)" \
    --volume "$HOME:$HOME" \
    --volume "$PWD:/mnt/project-root" \
    --workdir /mnt/project-root \
    --tty --interactive \
    --init \
    hseeberger/scala-sbt:8u141-jdk_2.12.3_0.13.16 \
    "$@"
