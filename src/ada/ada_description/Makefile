OR_URDF_LOAD=$(shell rospack find or_urdf)/scripts/load.py
OR_URDF_CONFIG=./config/herb_kinbody.yaml
PARAMS_POSTPROCESS=./scripts/postprocess_params.py
XACRO_POSTPROCESS=./scripts/postprocess_xacro.py
RM=rm -f

TARGETS=robots/mico.urdf ordata/robots/mico.robot.xml
#COMPONENTS=robots/herb_base.urdf.xacro robots/bh280.urdf.xacro robots/wam.urdf.xacro
PACKAGE=ada_description
#COLLISION_PRIMS=config/collision_primitives.json

.PHONY: all clean
.SECONDARY:

all: $(TARGETS)

clean:
	$(RM) $(TARGETS) $(COMPONENTS)

ordata/robots/mico.robot.xml: robots/mico.urdf
	$(OR_URDF_LOAD) $< $@ --config=$(OR_URDF_CONFIG)


robots/mico.urdf: $(COMPONENTS)

# Wrap the URDF in an xacro macro.
%.urdf.xacro: %_raw.urdf
	$(XACRO_POSTPROCESS) --name=$(notdir $*) --package=$(PACKAGE) --collision_meshes True $< $@


# General rules.
%: %.xacro
	rosrun xacro xacro.py $< > $@
