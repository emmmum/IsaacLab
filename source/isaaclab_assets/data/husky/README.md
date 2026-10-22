# Clearpath Husky

`husky.urdf` and `meshes/` come from Clearpath Robotics' `husky_description` package
(https://github.com/husky/husky, branch `noetic-devel`, BSD license, see `LICENSE`). The xacro was expanded with the
default arguments (no accessories, no top-plate extras), mesh paths were made relative and the Gazebo plugins were
removed. `usd/` was generated from it with

    ./isaaclab.sh -p scripts/tools/convert_urdf.py source/isaaclab_assets/data/husky/husky.urdf \
        source/isaaclab_assets/data/husky/usd --fix-base --joint-target-type none

The top plate is 0.3773 m above the floor; `base_link` is 0.1323 m above the floor.
