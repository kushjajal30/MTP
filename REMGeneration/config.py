__NDEM__ = 4
__NCPUS__ = 25
__output_path = 'REMS_1609'

##Terrain Generator Params
__number_of_buildings__ = 9
__building_min_width__ = 10
__building_min_length__ = 10
__building_max_width__ = 40
__building_max_length__ = 40
__terrain_size__ = 64
__min_height__ = 15
__max_height__ = 40

#REM Generator Params
__Ht__ = [40]
__Hr__ = 1.5
__fGHz__ = 0.010
__K__ = 1.3333
__polar_radius__ = (__terrain_size__/2)*(2**0.5)
__polar_radius_points__ = __terrain_size__
__polar_angle__ = 360
__polar_order__ = 3
__signal_strength__ = 20