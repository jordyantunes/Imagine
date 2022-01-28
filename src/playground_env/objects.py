import pygame
import numpy as np
from typing import List

from src.playground_env.color_generation import sample_color
from src.playground_env.env_controller import EnvController


class Thing:
    def __init__(self, object_descr, object_id_int, params):
        """
        Main object class.
        
        Parameters
        ----------
        object_descr: dict
            Dict that specify some attributes of the object that need to be created.
        object_id_int: int
            id of the object in the scene.
        params: dict
            Dict with all environment parameters.
        """

        self.object_descr = object_descr
        self.object_id_int = object_id_int
        self.params = params
        self.min_max_sizes = params['min_max_sizes']
        self.admissible_attributes = params['admissible_attributes']
        self.adm_rel_attributes = [a for a in self.admissible_attributes if 'relative' in a]

        self.agent_size = params['agent_size']
        self.obj_size_update = params['obj_size_update']
        self.get_attributes_functions = params['extract_functions']['get_attributes_functions']
        self.img_path = params['img_path']

        # initialize object attributes.
        self.object_attributes = self.object_descr.copy()
        self.object_initial_attributes = dict(zip(sorted(self.object_descr.keys()), [[self.object_descr[k]] for k in sorted(self.object_descr.keys())]))
        # add relative attributes (will be filled later when all objects in the scene have been created)
        for a in self.adm_rel_attributes:
            self.object_initial_attributes[a] = []

        self.__finished_initialization = False

        # Initialize physical attributes of the object that will compose its features.
        self.rgb_code = None
        self.position = None
        self.size = None
        self.type = None
        self.grasped = False
        self.scene_objects:List[Thing] = []  # list of refs to other objects from the scene
        self.status = None # On/off

        # initialize values for the type, position color and size.
        self._get_type_encoding()
        self._sample_position()
        self._sample_color()
        self._sample_size()
        self._sample_status()
        self.initial_rgb_code = self.rgb_code.copy()

        # rendering
        self.view = False
        self.patch = None

        self.off_icon = None
        self.__finished_initialization = True

    def update_ref_to_scene_objects(self, scene_objects):
        """
        Add reference to other objects in the scene. Update attributes (relative ones especially).
        Parameters
        ----------
        scene_objects: list of Thing objects.

        """
        self.scene_objects = scene_objects
        for k in self.admissible_attributes:
            self._update_attribute(k)

    # Sample physical attributes of the object
    def _sample_color(self):
        """
        Sample a color for the object.

        """
        if 'colors' in self.admissible_attributes:
            if 'shades' in self.admissible_attributes:
                rgb_code = sample_color(color=self.object_initial_attributes['colors'][0],
                                        shade=self.object_initial_attributes['shades'][0])
            else:
                rgb_code = sample_color(color=self.object_initial_attributes['colors'][0],
                                        shade=np.random.choice(['light', 'dark']))
        else:
            rgb_code = np.random.uniform(-1, 1, 3)
        self._update_color(rgb_code)
        

    def _sample_size(self):
        """
        Sample a size for the object

        """
        if 'sizes' in self.admissible_attributes:
            if self.object_initial_attributes['sizes'][0] == 'small':
                size = np.random.uniform(self.min_max_sizes[0][0], self.min_max_sizes[0][1])
            elif self.object_initial_attributes['sizes'][0]  == 'big':
                size = np.random.uniform(self.min_max_sizes[1][0], self.min_max_sizes[1][1])
            else:
                raise NotImplementedError
        else:
            size = np.random.uniform(self.min_max_sizes[0][0], self.min_max_sizes[1][1])
        self._update_size(size)

    def _sample_position(self):
        """
        Sample a position for the object.
        """
        lows = (-1, -1)
        highs = (1, 1)
        if 'positions' in self.admissible_attributes:
            if self.object_initial_attributes['positions'] == 'left':
                highs = (0, 1)
            elif self.object_initial_attributes['positions'] == 'right':
                lows = (0, -1)
            elif self.object_initial_attributes['positions'] == 'top':
                lows = (-1, 0)
            elif self.object_initial_attributes['positions'] == 'bottom':
                highs = (1, 0)
            else:
                raise NotImplementedError
        ok = False
        while not ok:
            candidate_position = np.random.uniform(lows, highs)
            ok = True
            for obj in self.scene_objects:
                if obj.position is not None:
                    if np.linalg.norm(obj.position - candidate_position) < self.params['epsilon_initial_pos']:
                        ok = False
            if ok:
                self._update_position(candidate_position)

    def get_obj_size_string(self):
        return 'small' if self.size < self.min_max_sizes[0][1] else 'big'

    def _sample_status(self):
        """
        Sample a status for the object

        """
        if 'status' in self.admissible_attributes:
            if self.object_initial_attributes['status'][0] == 'on':
                status = np.array([1])
            elif self.object_initial_attributes['status'][0]  == 'off':
                status = np.array([0])
            else:
                raise NotImplementedError
        else:
            print("Status not admissible")
            status = np.array([0])
        self._update_status(status)

    # getter
    @property
    def rgb_code(self) -> tuple:
        # print("RGB: ", self.__rgb_code)
        if self.__finished_initialization and not self.is_light_on() and self.__rgb_code is not None:
            # print("Returning dimmed color")
            # print("RGB: ", self.__rgb_code * 0.5)
            return self.__rgb_code * 0.5
        return self.__rgb_code

    @rgb_code.setter
    def rgb_code(self, rgb_code: np.array):
        self.__rgb_code = rgb_code

    def is_light_on(self) -> bool:
        for o in self.scene_objects:
            if isinstance(o, Light) and o.status == np.array([1]):
                return True
        return False

    # Update physical attributes of the object
    def _update_status(self, new_status):
        self.status = new_status
        self._update_attribute('status')

    def _update_size(self, new_size):
        self.size = new_size
        self.size_pixels = int(self.params['ratio_size'] * self.size)
        self._update_attribute('sizes')
        if self.scene_objects:
            for obj in self.scene_objects:
                obj._update_attribute('relative_sizes')
                
    def _update_color(self, new_rgb):
        self.rgb_code = new_rgb
        self._update_attribute('colors')
        self._update_attribute('shades')
        if self.scene_objects:
            for obj in self.scene_objects:
                obj._update_attribute('relative_colors')
                obj._update_attribute('relative_shades')

    def _update_position(self, new_position):
        clipped_position = np.clip(new_position, -1.2, 1.2)
        self.position = clipped_position.copy()
        self._update_attribute('positions')
        if self.scene_objects:
            for obj in self.scene_objects:
                obj._update_attribute('relative_positions')

    def _update_attribute(self, attribute):
        """
        Update a symbolic attribute as a function of other objects in the scene
        Parameters
        ----------
        attribute: str
            Attribute to be updated
        """
        all_attributes = [k for k in self.admissible_attributes if attribute in k]
        if len(self.scene_objects) > 0:
            object_features = [o.get_features() for o in self.scene_objects]
            for att in all_attributes:
                self.object_attributes[att] = self.get_attributes_functions[att](object_features, self.object_id_int)

    # Get type one hot code
    def _get_type_encoding(self):
        self.type = np.zeros([self.params['nb_types']])
        self.type[self.params['attributes']['types'].index(self.object_initial_attributes['types'][0])] = 1


    def _is_hand_over(self, agent_position):
        return np.linalg.norm(self.position - agent_position) < (self.size + self.agent_size) / 2

    def enforce_relative_attributes(self):
        """
        When the object needs to have some relative attribute (e.g. the darkest), sample the corresponding physical property until it is.
        """
        oks = [False for _ in range(3)]
        counter = 0
        while not all(oks) and counter < 100:
            if 'relative_positions' in self.admissible_attributes:
                if self.object_attributes['relative_positions'] != self.object_initial_attributes['relative_positions']:
                    self._sample_position()
                else:
                    oks[0] = True
            else:
                oks[0] = True
            if 'relative_shades' in self.admissible_attributes:
                if self.object_attributes['relative_shades'] != self.object_initial_attributes['relative_shades']:
                    self._sample_color()
                else:
                    oks[1] = True
            else:
                oks[1] = True
            if 'relative_sizes' in self.admissible_attributes:
                if self.object_attributes['relative_sizes'] != self.object_initial_attributes['relative_sizes']:
                    self._sample_size()
                else:
                    oks[2] = True
            else:
                oks[2] = True
            counter += 1
        return self.assert_equal_attributes(self.object_initial_attributes, self.object_attributes)


    def assert_equal_attributes(self, att_1, att_2):
        """
        Function to check that the required attributes are included in the current attributes of the object.
        Parameters
        ----------
        att_1: dict
            Dict of attributes (current attributes).
        att_2: dict
            Dict of attributes (target attributes).

        """
        # assert that attributes 1 and included in attributes 2
        assert sorted(att_1.keys()) == sorted(att_2.keys())
        for k in att_1.keys():
            for v in att_1[k]:
                if att_2[k] is not None:
                    if v not in att_2[k] :
                        return False
        return True

    def update_state(self, agent_position, gripper_state, objects, object_grasped, action):
        """
        Update the state of the object after interactions with the agent or other object.
        Parameters
        ----------
        agent_position: nd.array of size 2
            New agent position (2D).
        gripper_state: Boolean
            New gripper state (closed or open).
        objects: list of Thing objects
            Ref to other scene objects.
        object_grasped: Boolean
            Whether the object was grasped before.
        action: nd.array
            The action of the agent.

        """
        update_object_grasped = False

        # if the hand is close enough
        if self._is_hand_over(agent_position):
            # if not object is grasped, check if this one is being grasped
            if gripper_state and not object_grasped:
                self.grasped = True
                update_object_grasped = True
        
        # if an object is grasped
        # check if it's that one, if it's still grasped
        if object_grasped and self.grasped and not gripper_state:
            update_object_grasped = False
            self.grasped = False

        # if grasped, the object follows the hand
        if self.grasped:
            self._update_position(agent_position.copy())
            update_object_grasped = True

        self.features = self.get_features()
        return update_object_grasped

    def get_features(self):
        """
        Form features of the object.
        """
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        features = np.concatenate([self.type, self.position, np.array([self.size]), self.rgb_code, grasped_feature, self.status])
        return features

    def _color_surface(self, surface, rgb):
        arr = pygame.surfarray.pixels3d(surface)
        arr[:, :, 0] = rgb[0]
        arr[:, :, 1] = rgb[1]
        arr[:, :, 2] = rgb[2]

    def get_pixel_coordinates(self, xpos, ypos):
        return ((xpos + 1) / 2 * (self.params['screen_size'] * 2 / 3) + 1 / 6 * self.params['screen_size']).astype(np.int), \
               ((-ypos + 1) / 2 * (self.params['screen_size'] * 2 / 3) + 1 / 6 * self.params['screen_size']).astype(np.int)

    def update_rendering(self, viewer):
        x, y = self.get_pixel_coordinates(self.position[0], self.position[1])
        left = int(x - self.size_pixels // 2)
        top = int(y - self.size_pixels // 2)

        color = tuple(self.rgb_code * 255)
        # pygame.draw.rect(self.icon, color, (0, 0, self.size_pixels - 1, self.size_pixels - 1), 6)
        self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 
        self.surface = self.off_icon.copy() if self.status == 0 and self.off_icon else self.icon.copy()
        self._color_surface(self.surface, color)
        viewer.blit(self.surface,(left,top))
        # viewer.blit(self.icon, (left, top))

    def __repr__(self):
        msg = '\n\nOBJ #{}: '.format(self.object_id_int)
        for att in self.object_attributes:
            msg += '\n\t{}: {}'.format(att, self.object_attributes[att])
        return msg


class UsedSupply:
    obj: Thing
    supply: Thing

    def __init__(self, obj: Thing, supply: Thing) -> None:
        self.obj = obj
        self.supply = supply

    def __repr__(self):
        return "Item({}, {})".format(str(self.obj), str(self.supply))

    def __eq__(self, other):
        if isinstance(other, UsedSupply):
            return ((self.obj.object_id_int == other.obj.object_id_int) and (self.supply.object_id_int == other.supply.object_id_int))

    def __hash__(self):
        return hash(self.__repr__())


class LivingThings(Thing):
    def __init__(self,  object_descr, object_id_int, params):
        super().__init__( object_descr, object_id_int, params)


class Animals(LivingThings):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)

    def update_state(self, hand_position, gripper_state, objects, object_grasped, action):
        """
        Animal objects can be grown. This function checks whether a supply is put in contact with the animal. If it is, the animal grows.

        """
        # check whether water or food is close
        for obj in objects:
            if obj.object_descr['categories'] == 'supply':
                # check distance
                if np.linalg.norm(obj.position - self.position) < (self.size + obj.size) / 2:
                    # check action
                    size = min(self.size + self.obj_size_update, self.min_max_sizes[1][1] + self.obj_size_update)
                    self._update_size(size)

        return super().update_state(hand_position, gripper_state, objects, object_grasped, action)


class Furnitures(Thing):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


class Plants(LivingThings):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
       

    def update_state(self, hand_position, gripper_state, objects:List[Thing], object_grasped, action):
        """
        Plant objects can be grown. This function checks whether a water object is put in contact with the plant. If it is, the plant grows.

        """
        controller = EnvController.getInstance()

        if self.is_light_on():
            for obj in objects:
                if obj.object_descr['types'] == 'water':
                    if UsedSupply(self, obj) in controller.env.used_supplies:
                        # print("Supply already used, skip")
                        continue
                    # check distance
                    if np.linalg.norm(obj.position - self.position) < (self.size + obj.size) / 2:
                        # check action
                        size_update = self.obj_size_update
                        if obj.get_obj_size_string() == 'small':
                            size_update /= 2
                        size = min(self.size + size_update, self.min_max_sizes[1][1] + size_update)
                        self._update_size(size)
                        controller.env.add_used_supply(self, obj)
        return super().update_state(hand_position, gripper_state, objects, object_grasped, action)


class Supplies(Thing):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


# # # # # # # # # # # # # # # # # #
# Animals
# # # # # # # # # # # # # # # # # #

class Dog(Animals):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'dog.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Cat(Animals):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'cat.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Human(Animals):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'human.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Fly(Animals):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'fly.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Parrot(Animals):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'parrot.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Mouse(Animals):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'mouse.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Lion(Animals):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'lion.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Pig(Animals):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'pig.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Cow(Animals):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'cow.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Cameleon(Animals):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'cameleon.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


# # # # # # # # # # # # # # # # # #
# Plants
# # # # # # # # # # # # # # # # # #

class Cactus(Plants):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'cactus.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Rose(Plants):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'rose.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Grass(Plants):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'grass.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Bonsai(Plants):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'bonsai.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Algae(Plants):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'algae.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Carnivorous(Plants):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'carnivorous.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Tree(Plants):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'tree.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Bush(Plants):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'bush.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Tea(Plants):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'tea.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Flower(Plants):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'flower.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 

# # # # # # # # # # # # # # # # # #
# Furniture
# # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # #
# Light
# # # # # # # # # # # # # # # # # #

class Light(Thing):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        self.off_icon = 1
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'lightbulb.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels))
            self.off_icon = pygame.image.load(self.img_path + 'lightbulb-off.png')
            self.off_icon = pygame.transform.scale(self.off_icon, (self.size_pixels, self.size_pixels))
        # self._update_status(np.array([1]))
        # self._update_position(np.array([0, 0]))

class ActionableFurniture(Furnitures):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)

    def update_state(self, hand_position, gripper_state, objects, object_grasped, action):
        """
        Lamps objects can be grown. This function checks whether a water object is put in contact with the plant. If it is, the plant grows.

        """
        if self._is_hand_over(hand_position) and gripper_state and not object_grasped:
            self._update_status(np.array([0]) if self.status == np.array([1]) else np.array([1]))
            light_object = None
            if self.scene_objects:
                lights_on = False
                for o in self.scene_objects:
                    if isinstance(o, ActionableFurniture) and o.status == np.array([1]):
                        lights_on = True
                    elif isinstance(o, Light):
                        light_object = o
                if light_object:
                    light_object._update_status(np.array([1]) if lights_on else np.array([0]))
                


class Chair(Furnitures):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'chair.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Sofa(Furnitures):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'sofa.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Sink(Furnitures):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'sink.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Window(ActionableFurniture):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'window.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Carpet(Furnitures):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'carpet.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Cupboard(Furnitures):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'cupboard.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Desk(Furnitures):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'desk.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Lamp(ActionableFurniture):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'lamp.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels))


class Door(ActionableFurniture):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'door.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Table(Furnitures):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'table.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 

# # # # # # # # # # # # # # # # # #
# Supply
# # # # # # # # # # # # # # # # # #


class Water(Supplies):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'water.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 


class Food(Supplies):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params['render_mode']:
            self.icon = pygame.image.load(self.img_path + 'food.png')
            self.icon = pygame.transform.scale(self.icon, (self.size_pixels, self.size_pixels)) 



obj_type_to_obj = dict(dog=Dog,
                       cat=Cat,
                       chameleon=Cameleon,
                       human=Human,
                       fly=Fly,
                       parrot=Parrot,
                       mouse=Mouse,
                       lion=Lion,
                       pig=Pig,
                       cow=Cow,
                       cactus=Cactus,
                       carnivorous=Carnivorous,
                       flower=Flower,
                       tree=Tree,
                       bush=Bush,
                       grass=Grass,
                       algae=Algae,
                       tea=Tea,
                       rose=Rose,
                       bonsai=Bonsai,
                       door=Door,
                       chair=Chair,
                       desk=Desk,
                       lamp=Lamp,
                       table=Table,
                       cupboard=Cupboard,
                       sink=Sink,
                       window=Window,
                       sofa=Sofa,
                       carpet=Carpet,
                       food=Food,
                       water=Water,
                       light=Light)



def generate_objects(objects_descr, params):
    """
    From a list of desired objects and their attributes, generate the scene.
    Parameters
    ----------
    objects_descr: list of dict
        List of dict that describes the attributes of the desired objects.
    params: dict
        Environment parameters.

    Returns
    -------
    objs: list of Thing objects
        List of Thing objects, objects that will be included in the scene.
    """
    for o in objects_descr:
        assert o['types'] in obj_type_to_obj.keys(), "The object '{}' is not registered in the obj_type_to_obj dict".format(o['types'])

    objs = [obj_type_to_obj[o['types']](o, o_id_int, params) for o, o_id_int in zip(objects_descr, range(len(objects_descr)))]

    # give each object the reference to other objects in the scene and resample their position so that they are not in contact.
    for o in objs:
        o.update_ref_to_scene_objects(objs)
        o._sample_position()

    # Enforce that relative attributes are respected (the darkest object should be the darkest physically).
    oks = [False for _ in range(len(objs))]
    while not all(oks):
        for i_o, o in enumerate(objs):
            oks[i_o] = o.enforce_relative_attributes()
    return objs


stop = 1
