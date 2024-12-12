"""
Runs the basic VSL experiment with added random walker tracking.

FEATURES:
Exception handling.
Logging.

FLOW:
Initialization
    *With no screen just CLI.*
    
    Ask session and subject ID.
    
    Set experiment and phase (and subphase if applicable): E_P or E_P_P
        In first iteration: E = 1, P = 1.
    
    Load inputs accordingly.

Run specified version of exp:
    *With screen.*
    
    Show instructions.
    
    Run E_1 with added tracking task.
        E_1: show succession of scenes assembled according to keyfile.
        Tracking task:
    
    Save data incrementally at each frame.

Exit:
    *With screen.*
    Show goodbye screen then timed exit.
"""

def draw_familiarization_trial_or_intertrial():
    global is_intertrial, fps
    screen.fill((255, 255, 255))
    if not is_intertrial:
        draw_shapes(
        screen,
        x=window_size[0]//2, y=window_size[1]//2,
        cell_size=shape_size, 
        shapes_data = shapes_data, 
        shapes_to_draw = trial_data['shapes'],
        positions = trial_data['positions']
        )
    draw_grid(
        screen,
        n=cell_num,
        x=window_size[0]//2, y=window_size[1]//2,
        cell_size=shape_size
        )
    draw_blended_noise_mask(
        screen, 
        x=window_size[0]//2, y=window_size[1]//2, 
        mask=noise_mask_1
        )
    draw_gaussian_blob(screen, tracked_blob, target_pos[0], target_pos[1])
    draw_blended_noise_mask(
        screen, 
        x=window_size[0]//2, y=window_size[1]//2, 
        mask=noise_mask_2
        )
    draw_mouse(screen, mouse_pos)
    pygame.display.flip()

    clock.tick(fps) # Hz

def draw_test_trial_or_intertrial():
    global is_intertrial, test_phase_trial_stage, fps

    if is_intertrial:
        screen.fill((125, 125, 125))

        draw_text(screen,'Melyik pár volt az ismerősebb?', offset = (0,-150),font = pygame.font.Font(pygame.font.match_font("arial"), int(60*font_size_corrector)))
        draw_text(screen,'Ha az első, akkor nyomd meg az 1-es gombot,', offset = (0,125),font = pygame.font.Font(pygame.font.match_font("arial"), int(60*font_size_corrector)))
        draw_text(screen,'ha a második, akkor pedig az 2-es gombot!', offset = (0,175),font = pygame.font.Font(pygame.font.match_font("arial"), int(60*font_size_corrector)))
    else:
        screen.fill((255, 255, 255))
        if test_phase_trial_stage in (1,3):
            if test_phase_trial_stage == 1:
                pair_num = trial_data['first_pres']
            elif test_phase_trial_stage == 3:
                pair_num = 0 if trial_data['first_pres'] == 1 else 1
            #print(f"pair_num: {pair_num}, shapes: {trial_data['shapes']}")
            draw_shapes(
                screen,
                x=window_size[0]//2, y=window_size[1]//2,
                cell_size=shape_size, 
                shapes_data = shapes_data, 
                shapes_to_draw = trial_data['shapes'][pair_num-1],
                positions = trial_data['positions'][pair_num-1]
                )
        draw_grid(
            screen,
            n=cell_num,
            x=window_size[0]//2, y=window_size[1]//2,
            cell_size=shape_size
            )
        draw_blended_noise_mask(
            screen, 
            x=window_size[0]//2, y=window_size[1]//2, 
            mask=noise_mask_1
            )
    
    #draw_mouse(screen, mouse_pos)
    pygame.display.flip()

    clock.tick(fps) # Hz

def increment_stage():
    global stage, data_file_path, verbose_logging

    if stage == 'instructions':
        stage = 'trials'
    elif stage == 'trials':
        if all_trials_shown():
            stage = 'thanks'
        else:
            stage = 'rest'
    elif stage == 'rest':
        stage = 'trials'

def all_trials_shown():
    global trial,trial_num
    if trial >= trial_num:
        return True
    else:
        return False

if __name__ == "__main__":
    try:
        """
        Run all code here.
        """
        #region Initialization
        from utils.codebase import *

        dev = False
        verbose_logging = True

        #region Set session data
        if dev:
            series = 'dev'
            subject_ID = '1'
            exp = '1'
            phase = '1'
            familiarization_trial_repetition = 1
            subphase = ''
        else:
            # Get Series and Subject ID
            series = input('Series: ')
            subject_ID = input('Subject ID: ')
            exp = '1'
            #phase = '1'
            familiarization_trial_repetition = -1
            subphase = ''

            ## Set run code to determine which type of session to run
            ## Exp1: familiarization: 11, test: 12
            ## Exp2: familiarization: 21, test: 22
            ## Exp3: familiarization: 31, pairs test: 321, singles test: 322
            #while True:
            #    exp = input('Experiment (1 - exp1, 2 - exp2, 3 - exp3): ')
            #    if exp in ('1', '2','3'):
            #        break
            while True:
                phase = input('Phase: (1 - familiarization, 2 - test): ')
                if phase in ('1', '2'):
                    break
            ## Set repetition of familiarization phase
            #if phase == '1':
            #    while True:
            #        familiarization_trial_repetition = input('How many repetitions for the familiarization trials? ')
            #        if familiarization_trial_repetition.isdigit():
            #            familiarization_trial_repetition = int(familiarization_trial_repetition)
            #            break
            #else:
            #    familiarization_trial_repetition = -1
            ## Set subphase for Exp3
            #if phase == '2' and exp == '3':
            #    while True:
            #        subphase = input('Pairs or singles (1 - pairs, 2 - singles): ') if (exp == '3' and phase == '2') else ''
            #        if subphase in ('1', '2'):
            #            break
            #else:
            #    subphase = ''

        random_pres = True

        session_code = f"{exp}{phase}{subphase}"
        #endregion

        #region Load inputs and prepare outputs
        # Load keyfile and decoder files
        input_directory = f'{os.getcwd()}/inputs/'
        scenes = np.genfromtxt(input_directory+'keyfile_'+session_code+'.csv', delimiter=',', dtype=int)
        word_decode = np.genfromtxt(input_directory+'worddecode_'+session_code+'.csv', delimiter=',', dtype=int)
        nonword_decode = np.genfromtxt(input_directory+'nonworddecode_'+session_code+'.csv', delimiter=',', dtype=int) if phase == '2' else ''

        # Load instructions
        with open(input_directory+'instructions_'+session_code+'.txt', 'r', encoding='utf-8') as file:
            instructions = file.read()

        # Set grid dimensions
        viewing_dist = 75 # cm
        used_monitor_no = 0 # The one the experiment runs on
        primary_monitor_no = 0 # The one treated as primary by the OS (on Win this has the detailed start menu for example)
        ppc = get_monitor_ppcs()[used_monitor_no] # Pixel density of used monitor in pixels per centimeters
        font_size_corrector = ppc/63 # So that the text shown will be the same relative size as on the dev machine
        shape_size_in_visual_angles = 0.1
        shape_size = visual_angles_to_pixels(shape_size_in_visual_angles, ppc, viewing_dist)
        cell_num = 3

        #region Set up screen
        """Needed for image loading."""
        pygame.init()
        if dev:
            window_size = (shape_size * (cell_num+0.5),shape_size * (cell_num+0.5))
            screen = pygame.display.set_mode(window_size)
        else:
            window_size = pygame.display.get_desktop_sizes()[primary_monitor_no]
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("Visual Statistical Learning with Target Tracking")
        clock = pygame.time.Clock()
        #endregion

        # Load shapes
        shapes_data = []
        for i, file in enumerate(sorted([f for f in os.listdir('inputs/shapes') if f.endswith('.png')], key=lambda x: int(x.split('.')[0].replace('shape', '')))):
            img_path = os.path.join('inputs/shapes', file)
            img =  pygame.image.load(img_path).convert()
            shapes_data.append(pygame.transform.scale(img, (shape_size, shape_size)))
        
        # Create output directory
        output_directory = f'{os.getcwd()}/outputs/'
        subject_directory = output_directory+series+'/'+str(subject_ID)+'/'
        session_directory = output_directory+series+'/'+str(subject_ID)+'/'+session_code+'/'

        if not os.path.exists(session_directory):
            os.makedirs(session_directory)
        else:
            if dev:
                overwrite = 'y'
            else:
                while True:
                    overwrite = input('Subject & session directory exists for this series. Overwrite? (y/n) ').lower()
                    if overwrite in ('y', 'n'):
                        break

            if overwrite == 'n':
                print('Aborting due to already existing run for subject-session pair in this series.')
                quit()
        
        if phase == '1':  # generate & save shape mapping
            if not dev and os.path.exists(session_directory+'/shape_mapping.csv'):
                shape_mapping = np.loadtxt(session_directory+'/shape_mapping.csv', delimiter=',').astype(np.int64)
                np.savetxt(session_directory+'shape_mapping.csv', shape_mapping, delimiter=',', fmt='%d')
            else:
                shape_mapping = np.random.permutation(np.arange(12))+1
                np.savetxt(session_directory+'shape_mapping.csv', shape_mapping, delimiter=',', fmt='%d')
        elif phase == '2':  # load previously used soundmapping from familiarization directory
            try:
                shape_mapping = np.loadtxt(subject_directory+'/'+exp+'1/shape_mapping.csv', delimiter=',').astype(np.int64)
            except FileNotFoundError:
                print('Shape mapping file not found in this series. Please run a familiarization session first for this subject.')
                quit()

        if random_pres:
            scene_order = np.random.permutation(np.arange(scenes.shape[0]))+1
        else:
            scene_order = np.arange(scenes.shape[0])+1
        if 1 < familiarization_trial_repetition:
            scene_order = np.repeat(scene_order, familiarization_trial_repetition)
        np.savetxt(session_directory+'scene_order.csv', scene_order, delimiter=',', fmt='%d')

        trial_num = len(scene_order)
        trial = 0

        if phase == '2':
            subject_keypresses = np.full(trial_num, -1)
            first_pres_list = np.full(trial_num, -1)

        if phase == '1':
            data_file_path = set_up_logging(session_directory, 'data')
            log("Time,Block,Trial,Scene_from_keyfile,Shapes,Positions,Tracked_x,Tracked_y,Mouse_x,Mouse_y",data_file_path,verbose_logging)
        #endregion

        stage = 'instructions'      

        #endregion

        #region Experiment settings
        trial_duration = 3 if phase == '1' else 1.5 # In secs
        intertrial_duration = 1 if phase == '1' else 0.25 # In secs
        if series == 'test':
            trial_duration /= 10
            intertrial_duration /= 10
        trial_num_per_block = 18 # (3+1)*144 = 576's divisors: [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 64, 72, 96, 144, 192, 288, 576]
        
        random_start_pos_for_target = False
        fps = 60 # Hz
        std_pos_in_visual_angles = 0.04
        std_vel_in_visual_angles = 0
        blob_size_in_visual_angles = 0.05
        boundary_type = 'reflective'
        #boundary_type = 'clip'
    
        settings_to_save = {
            'trial_duration':trial_duration,
            'intertrial_duration':intertrial_duration,
            'trial_num':trial_num,
            'trial_num_per_block':trial_num_per_block,
            'random_start_pos_for_target':random_start_pos_for_target,
            'shape_size_in_visual_angles':shape_size_in_visual_angles,
            'std_pos_in_visual_angles':std_pos_in_visual_angles,
            'std_vel_in_visual_angles':std_vel_in_visual_angles,
            'blob_size_in_visual_angles':blob_size_in_visual_angles,
            'boundary_type': boundary_type,
            'fps':fps,
            'dt':1 / fps,
            'viewing_distance':viewing_dist,
            'pixel_per_cm_of_monitor':ppc
        }

        if phase == '1':
            with open(os.path.join(session_directory, 'settings.txt'), "w") as file:
                for key, value in settings_to_save.items():
                    file.write(f"{key}: {value}\n")

        window_in_focus = False
        mouse_clicked = False
        trials_finished_in_block = 0
        block = 0
        length_of_block = trial_num_per_block*(trial_duration + intertrial_duration)
        num_of_blocks = trial_num/trial_num_per_block

        #region Run trials
        last_switch = pygame.time.get_ticks()
        is_intertrial = True
        replied = False
        response = None  
        finished = False
        scene = None
        trial_data = {'shapes':None,'positions':None}
        test_phase_trial_stage = 0 #0: blank, 1: first pair, 2: blank, 3: second pair, 4: blank, all for 1 sec

        saved = False

        bounding_rectangle = { # Dimensions of the bounding rectangle as determined by the grid
                'x_min': window_size[0]//2 - shape_size * cell_num/2,
                'x_max': window_size[0]//2 + shape_size * cell_num/2,
                'y_min': window_size[1]//2 - shape_size * cell_num/2,
                'y_max': window_size[1]//2 + shape_size * cell_num/2
            }

        noise_mask_1 = generate_blended_noise_mask(
                    width=window_size[0], height=window_size[1],
                    noise_strength=255, noise_density=0.5, smoothing=0, blend_alpha=128
                    )
        
        noise_mask_2 = generate_blended_noise_mask(
            width=window_size[0], height=window_size[1],
            noise_strength=255, noise_density=0.5, smoothing=0, blend_alpha=50
            )

        tracked_blob = generate_gaussian_blob(visual_angles_to_pixels(blob_size_in_visual_angles, ppc, viewing_dist), min_size=3)

        std_pos = visual_angles_to_pixels(std_pos_in_visual_angles, ppc, viewing_dist)
        std_vel = visual_angles_to_pixels(std_vel_in_visual_angles, ppc, viewing_dist)
        dt = 1 / fps

        target = Target(
            [np.random.uniform(bounding_rectangle['x_min'], bounding_rectangle['x_max']),np.random.uniform(bounding_rectangle['y_min'], bounding_rectangle['y_max'])]
            if random_start_pos_for_target else
            [window_size[0]//2,window_size[1]//2],
            std_pos = std_pos,
            std_vel = std_vel,
            dt = dt
            )

        pygame.mouse.set_visible(False)
        
        while not finished:
            for event in pygame.event.get(): # The pygame.event.get() call is needed to handle the event queue
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.ACTIVEEVENT:
                    # Check focus state
                    window_in_focus = bool(event.gain)  # True if window gained focus, False if lost
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_clicked = True
                elif event.type == pygame.KEYDOWN:  # Check for key presses
                    if event.key == pygame.K_1:  # Check if the '1' key is pressed
                        response = 1
                        subject_keypresses[trial-1] = response
                        replied = True
                    elif event.key == pygame.K_2:  # Check if the '2' key is pressed
                        response = 2
                        subject_keypresses[trial-1] = response
                        replied = True

            mouse_pos = pygame.mouse.get_pos()
            if window_in_focus:
                if stage == 'instructions':
                    if phase == '1': # Familiarization phase
                        screen.fill((125, 125, 125))
                        draw_text(screen,'A most következő kísérletben egy mozgó fehér foltot fogsz látni egy háttér előtt.', offset = (0,-450),font = pygame.font.Font(pygame.font.match_font("arial"), int(60*font_size_corrector)))
                        draw_text(screen,'A feladat, hogy a mozgó fehér folt közepét a lehető legpontosabban kövesd az egérrel.', offset = (0,-350),font = pygame.font.Font(pygame.font.match_font("arial"), int(60*font_size_corrector)))
                        draw_text(screen,'Szólj a kísérletvezetőnek, aki szóban is el fogja mondani a feladatot, utána tudod elkezdeni.', offset = (0,-250),font = pygame.font.Font(pygame.font.match_font("arial"), int(60*font_size_corrector)))
                        draw_text(screen,'A kezdéskor a folt pontosan a képernyő közepén lesz.', offset = (0,-150),font = pygame.font.Font(pygame.font.match_font("arial"), int(60*font_size_corrector)))
                        starter_circle_radius = 30 #
                        pygame.draw.circle(screen, (255, 255, 255), (window_size[0]//2, window_size[1]//2), starter_circle_radius, width = 5)

                        draw_mouse(screen, mouse_pos)
                        pygame.display.flip()
                
                        if (mouse_pos[0] - window_size[0]//2) ** 2 + (mouse_pos[1] - window_size[1]//2) ** 2 <= starter_circle_radius ** 2 and mouse_clicked:
                            increment_stage()
                            mouse_clicked = False
                    else: # Test phase
                        screen.fill((125, 125, 125))
                        draw_text(screen,'Most ábrákat fogsz látni az előző részből, először egy ábrapárt, aztán egy másodikat.', offset = (0,-250),font = pygame.font.Font(pygame.font.match_font("arial"), int(60*font_size_corrector)))
                        draw_text(screen,'A feladatod, hogy megválaszold, melyiket találod ismerősebbnek az előző rész alapján.', offset = (0,-150),font = pygame.font.Font(pygame.font.match_font("arial"), int(60*font_size_corrector)))
                        draw_text(screen,'Szólj a kísérletvezetőnek, aki szóban is el fogja mondani a feladatot, utána tudod majd elkezdeni.', offset = (0,-50),font = pygame.font.Font(pygame.font.match_font("arial"), int(60*font_size_corrector)))
                        draw_mouse(screen, mouse_pos)
                        pygame.display.flip()
                
                        if mouse_clicked:
                            increment_stage()
                            mouse_clicked = False
                elif stage == 'trials':
                    if phase == '1': # Familiarization phase
                        #region Update target state
                        
                        # Iterate target position based on state transition equation
                        target.walk()
                        target_pos = target.get_pos()
                        if boundary_type == 'reflective':
                            target.apply_reflective_boundaries(**bounding_rectangle)
                        elif boundary_type == 'clip':
                            target.apply_hard_boundaries(**bounding_rectangle)
                        elif boundary_type == 'potential':
                            # Compute the force due to the potential at the current position
                            force = compute_force_due_to_repulsive_potential(
                                target_pos,
                                k = 100.0, # Potential strength
                                alpha = 2.0, # Potential steepness
                                **bounding_rectangle
                                )
                            # Apply force to walker
                            target.apply_forces(force)
                            target.dissipate_kinetic_energy(dampening = 0.99)
                            #print(f"Pos: {target.get_pos()}, vel: {target.get_vel()}")

                        #endregion
                        
                        #region Draw
                        screen.fill((255, 255, 255))
                        draw_familiarization_trial_or_intertrial()
                        #endregion
                        
                        #region Update iteration
                        if is_intertrial:
                            if intertrial_duration <= (pygame.time.get_ticks() - last_switch)/1000:
                                is_intertrial = False
                                trial += 1
                                trials_finished_in_block += 1
                                scene = scene_order[trial-1]
                                trial_data = familiarization_trial(scenes, scene, shape_mapping, word_decode)
                                last_switch = pygame.time.get_ticks()
                        else:
                            if trial_duration <= (pygame.time.get_ticks() - last_switch)/1000:
                                is_intertrial = True
                                last_switch = pygame.time.get_ticks()

                        #region Stage management
                        if trials_finished_in_block == trial_num_per_block:
                            increment_stage()
                        #endregion
                        
                        #region Save data
                        log(
                            f"{pygame.time.get_ticks()},{block},{trial},{scene},{trial_data['shapes']},{trial_data['positions']},{int(target_pos[0])},{int(target_pos[1])},{mouse_pos[0]},{mouse_pos[1]}",
                            data_file_path,
                            verbose_logging
                            )
                        #endregion
                    else: # Test phase
                        #region Update iteration
                        if is_intertrial: # That is, reply
                            if replied or trial == 0: #DEV, alany válaszolt-e
                                is_intertrial = False
                                trial += 1
                                test_phase_trial_stage = 0
                                replied = False
                                response = None
                                scene = scene_order[trial-1]
                                trial_data = pairs_test_trial(scenes, scene, shape_mapping, word_decode, nonword_decode, True)
                                first_pres_list[trial-1] = trial_data['first_pres']
                                last_switch = pygame.time.get_ticks()
                        else: # That is, pair presentation
                            #print(f"test_phase_trial_stage = {test_phase_trial_stage}")
                            if test_phase_trial_stage in (1,3):
                                relevant_duration = trial_duration
                            else:
                                relevant_duration = intertrial_duration
                            if relevant_duration <= (pygame.time.get_ticks() - last_switch)/1000: # Based on timing
                                if test_phase_trial_stage < 4:
                                    test_phase_trial_stage += 1
                                else:
                                    is_intertrial = True
                                last_switch = pygame.time.get_ticks()
                        #endregion

                        #region Draw
                        draw_test_trial_or_intertrial()
                        #endregion
                        
                        #region Stage management
                        if all_trials_shown():
                            increment_stage()
                        #endregion
                elif stage == 'rest':
                    screen.fill((125, 125, 125))
                    draw_text(screen,'Most egy rövid szünetet következik.', offset = (0,-450),font = pygame.font.Font(pygame.font.match_font("arial"), int(60*font_size_corrector)))
                    draw_text(screen,'Engedd el az egeret és mozgasd át a kezed és az ujjaid.', offset = (0,-350),font = pygame.font.Font(pygame.font.match_font("arial"), int(60*font_size_corrector)))
                    draw_text(screen,'Ha eleget pihentél, klikkelj a kör közepébe a folytatáshoz!', offset = (0,-250),font = pygame.font.Font(pygame.font.match_font("arial"), int(60*font_size_corrector)))
                    draw_text(screen,'A kezdéskor a folt pontosan a képernyő közepén lesz.', offset = (0,-150),font = pygame.font.Font(pygame.font.match_font("arial"), int(60*font_size_corrector)))
                    starter_circle_radius = 30 #
                    pygame.draw.circle(screen, (255, 255, 255), (window_size[0]//2, window_size[1]//2), starter_circle_radius, width = 5)

                    draw_mouse(screen, mouse_pos)
                    pygame.display.flip()
            
                    if (mouse_pos[0] - window_size[0]//2) ** 2 + (mouse_pos[1] - window_size[1]//2) ** 2 <= starter_circle_radius ** 2 and mouse_clicked:
                        block += 1
                        trials_finished_in_block = 0
                        target.reset_pos()
                        mouse_clicked = False
                        increment_stage()
                elif stage == 'thanks':
                    if phase == '2' and not saved:
                        np.savetxt(session_directory+'subject_keypresses.csv',subject_keypresses.astype(int), delimiter=',', fmt = '%d')
                        np.savetxt(session_directory+'subject_responses.csv',(first_pres_list == subject_keypresses).astype(int), delimiter=',', fmt = '%d')
                        np.savetxt(session_directory+'first_pres_list.csv', first_pres_list.astype(int), delimiter=',', fmt = '%d')
                        print(f"Correct: {round_to(100*np.mean(first_pres_list == subject_keypresses),0.01)}%")
                        saved = True
                    screen.fill((125, 125, 125))
                    draw_text(screen,'Köszönjük!', offset = (0,-120),font = pygame.font.Font(pygame.font.match_font("arial"), int(60*font_size_corrector)))
                    draw_text(screen,'Vége a kísérlet első részének.' if phase == '1' else 'Vége a kísérletnek.', offset = (0,0),font = pygame.font.Font(pygame.font.match_font("arial"), int(60*font_size_corrector)))
                    draw_text(screen,'Kattints a kilépéshez.', offset = (0,120),font = pygame.font.Font(pygame.font.match_font("arial"), int(60*font_size_corrector)))
                    draw_mouse(screen, mouse_pos)
                    pygame.display.flip()
            
                    if mouse_clicked:
                        finished = True
        if finished:
            exit()
        #endregion

    except Exception as e:
        """
        Treat exceptions here.
        """
        traceback.print_exc()
    finally:
        """
        Export data here upon error.
        """
        pass
