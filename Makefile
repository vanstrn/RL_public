
define run_command
		docker run -it --rm --name raide-rl \
			--gpus all \
			--privileged \
			--network=host \
			-e USER=${USER} \
			-v ${PWD}:${PWD} \
			-v /tmp/.X11-unix:/tmp/.X11-unix \
			-v $$HOME:/home/${USER} \
			-v /etc/passwd:/etc/passwd \
			-w ${PWD} \
			raide-rl \
			$(1)
endef

all: ${PREREQ_STEPS} libs install

#Can't be the same name as a directory
image:
	docker build --rm -t raide-rl -f docker/raide-rl/Dockerfile docker/raide-rl
dev:
	$(call run_command,bash)
clean:
	rm -r images/ logs/ models/
#Ability to run unit tests
# test:

.PHONY: install build
