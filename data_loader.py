import os
import neo
import h5py
import warnings
import pandas as pd
import numpy as np
import scipy.io as sio
from collections import defaultdict

class DataLoader:
    def __init__(self, root, struct = 'BLA', time_ref = 'ms', cell_type = 1, rats = None, save = True):
        """
        Initialize the Data Loader.

        Parameters
        ----------
        root : str
            Path of the dataset root folder.
        struct : str (optional)
            Name of the structure to load. Keys to be used 'BLA', 'BLV', 'BMA', 'BMP', 
            'CeCM', 'CPu', 'DEn', 'GP', 'Hpc', 'LaDL', 'Pir', 'VEn', 'basal', 'olfact'.
            Default is BLA, it's possible to check in the map file.
        time_ref : str (optional)
            Time unit for spike timestamps ('ms' for milliseconds, 's' for seconds).
            Default is 'ms'.
        cell_type : int (optional)
            Cell type to be used, pyramidal as default (1 = pyramidal, 2 = interneurons).
        rats : list (optional)
            List to determine which rat/rats to load, default value is None to load all rats.
        save : bool (optional)
            Determine if will save or not the spikes and the LFP's, before load the data the
            code will check if the file already existis in Global_vars folders and will load it.

        Examples
        --------
        >>> root = '/path/to/data'
        >>> struct = 'BLA' # load BLA data
        >>> time_ref = 'ms' # work in milliseconds
        >>> cell_type = 1 # load just the pyramidal cells
        >>> rats = ['Rat08','Rat11'] # load just Rat 8 and 11
        >>> save = True # save lfp and spikes, if the file already exists, will be loaded

        Author
        -------
        written by Tulio Almeida 11/2023
        """
        # Assert conditions
        assert type(root) == str, 'root should be a path of the main folder /path/to/GG-Dataset-Tulio'
        assert type(struct) == str and struct in ['BLA', 'BLV', 'BMA', 'BMP', 
                                                  'CeCM', 'CPu', 'DEn', 'GP', 
                                                  'Hpc', 'LaDL', 'Pir', 'VEn', 
                                                  'basal', 'olfact'], "'struct' should be str and be in map file"
        assert type(time_ref) == str and time_ref in ['ms','s'], "'time_ref' should be str and be 'ms' or 's'"
        assert type(cell_type) == int and cell_type in [1,2], "'cell_type' must be 1 for pyr cell or 2 for interneurons"
        if rats is not None:
            for rat in rats:
                assert rat in ['Rat08','Rat09','Rat10','Rat11'], "You must choose between Rat08,Rat09,Rat10,Rat11"

        # Load global variables
        self.data_path = os.path.abspath(root)
        self.struct = struct
        self.structure = sio.loadmat(os.path.abspath(os.path.join(root,"Global_vars\\structures.mat")))[struct]
        self.final_type = sio.loadmat(os.path.abspath(os.path.join(root,"Global_vars\\finalType.mat")))['finalType']
        self.cell = cell_type
        self.fs = 1250 # Acquisition frequency
        self.rats = rats
        self.save = save

        # Initializing functions
        self.t_ref = self._get_t_unit(time_ref)
        self.paths = self._get_path()
        self.maps = self._get_maps()
        self.rat_idx = self._get_rats_idx()
        self.states = self._get_states()
        self.states_ordered = self._get_states_ordered()
        self.events = self._get_events()
        self.position = self._get_pos()

    def __warning_msg(self, file):
        """
        Issue a warning. This method issues a deprecation warning indicating that a 
        file does not exist in the dataset.

        Returns
        -------
        None
            This method does not return a value. It issues a warning instead.

        Author
        -------
        written by Tulio Almeida 11/2023
        """
        warnings.warn('The file does not exist in the dataset, check the folder \n %s'%file,stacklevel=2)

    def _get_t_unit(self,time_ref):
        """
        Get the time unit reference to use as reference in the DataLoader functions.

        Parameters
        ----------
        time_ref : str
            Time unit for spike timestamps ('ms' for milliseconds, 's' for seconds).
            Default is 'ms'.

        Returns
        -------
        int
            Constant value to multiply the time values.

        Author
        -------
        written by Tulio Almeida 11/2023
        """
        t_ref = 1 if time_ref == 's' else 1000

        return t_ref

    def _get_rats_idx(self):
        """
        Create a dicitionary that allow to loop rats using the str as reference (when using the folder 
        as reference; as in structures) or rat number as reference (when using the files from the
        GG-Dataset-Orirignal; as in 'finalType'). This function will filter which rats to load based 
        on the initialisation (rats variable).

        Returns
        -------
        dict
            A dictionary containing the rat number in str as key and in int as value.

        Author
        -------
        written by Tulio Almeida 11/2023
        """
        rat_idx = {'Rat08':8,'Rat09':9,'Rat10':10,'Rat11':11}
        
        # Return the rat dictionary filtering by the rats choosed in the initialisation
        if self.rats is None:
            return rat_idx
        else:
            rat_idx = {rat: rat_idx[rat] for rat in self.rats}
            return rat_idx
    
    def _get_path(self):
        """
        Get all the paths for all file for each rat and session in the given directory. This
        function doesn't check if the path exisist! This check is done inside each 'get' function.
        Thus, all possible paths will be created as default even if the file doens't exist in 
        one specific session.

        Parameters
        ----------
        root : str
            Path of the dataset root folder.

        Returns
        -------
        dict
            A dictionary containing all the paths for each rat and session.

        Author
        -------
        written by Tulio Almeida 11/2023
        """
        # Initialize an empty dictionary to store the subfolders, paths
        subfolders, paths_dict = {}, {}

        # Get the list of all files and directories in the directory
        items = os.listdir(self.data_path)

        # Loop through each item in the directory
        for item in items:
            # Check if the item is a directory
            if os.path.isdir(os.path.join(self.data_path, item)) and 'Rat' in item:
                # Add the directory to the dictionary
                subfolders[item] = os.path.join(self.data_path, item)

        # Iterate every rat
        for rat, path in subfolders.items():
            # Create the rat dict
            paths_dict[rat] = {}
            # Initialize the counter to represent the days (used with finalType)
            count = 1
            # Walk through the directory
            for dirpath, dirnames, filenames in os.walk(path):
                # If the current directory does not contain any subdirectories
                if not dirnames:
                    # Add the the paths to the dictionary
                    session_name = dirpath.split(os.sep)[-1]
                    paths_dict[rat][session_name] = {'spike':os.path.join(dirpath,(session_name + '-spikes.mat')),
                                                     'index':os.path.join(dirpath,(session_name + '-CellIndex.mat')),
                                                     'lfp':os.path.join(dirpath,(session_name + '.lfp')),
                                                     'position':os.path.join(dirpath,(session_name + '-pos.csv')),
                                                     'nrs':os.path.join(dirpath,(session_name + '.nrs')),
                                                     'RunTimes-mat':os.path.join(dirpath,(session_name+'-TrackRunTimes.mat')),
                                                     'reward':os.path.join(dirpath, 'reward.mat'),
                                                     'States':os.path.join(dirpath, 'States.mat'),
                                                     'Ripples-mat':os.path.join(dirpath, 'SWSripples.mat'),
                                                     'airpuff-mat':os.path.join(dirpath, 'airpuff.mat'),
                                                     'runintervals':os.path.join(dirpath, 'runintervals.mat'),
                                                     'lrw':os.path.join(dirpath,(session_name + '.lrw.evt')),
                                                     'rrw':os.path.join(dirpath,(session_name + '.rrw.evt')),
                                                     'rip':os.path.join(dirpath,(session_name + '.rip.evt')),
                                                     'puf':os.path.join(dirpath,(session_name + '.puf.evt')),
                                                     'cat':os.path.join(dirpath,(session_name + '.cat.evt')),
                                                     'day': count}
                    # Iterate the day
                    count +=1

        # Return the paths dictionary filtering by the rats choosed in the initialisation
        if self.rats is None:
            return paths_dict
        else:
            paths_dict = {rat: paths_dict[rat] for rat in self.rats}
            return paths_dict
    
    def _get_maps(self):
        """
        Generates a dictionary that mapss rats to their sessions and the corresponding clusters.
        This function initializes an empty dictionary to store the subfolders, and struct shanks. 
        It then lists all files and directories in the data path. For each item in the directory, 
        if the item is a directory and contains 'Rat', it adds the directory to the dictionary.
        It then iterates over every rat to create the structure to save the shanks for each side. 
        It creates a reference for rat name as key and rat number as value, and a reference dict 
        of left and right shanks threshold.

        For each rat, it iterates over each session. If the rat is 'Rat08' or 'Rat09', it uses the 
        'less equal' for left and greater for right. If the rat is 'Rat10' or 'Rat11', it uses the 
        'less equal' for right and greater for left. It fills the dictionary with the left and right 
        shanks. This is done based on how the mapss are organized, order of the values (check maps pdf).

        Finally, it iterates over every rat and removes the sessions that don't have 'structure' data and 
        remove this session from the final dict.

        Returns
        -------
        dict
            A dictionary that mapss rats to their sessions, and each session to the corresponding clusters.

        Examples
        --------
        >>> self.data_path = '/path/to/data'
        >>> self.maps = np.array([[0, 1, 1, 1, 1], [1, 1, 2, 2, 2], [2, 2, 3, 3, 3], [3, 3, 4, 4, 4]])
        >>> self._get_maps()
        {'Rat01': {'Session1': {'left': [1, 2], 'right': [3, 4]}, 'Session2': {'left': [5, 6], 'right': [7, 8]}}

        Author
        -------
        written by Tulio Almeida 11/2023
        """
        # Initialize an empty dictionary to store the subfolders, and maps shanks
        subfolders,maps = {},{}

        # Get the list of all files and directories in the directory
        items = os.listdir(self.data_path)

        # Loop through each item in the directory
        for item in items:
            # Check if the item is a directory
            if os.path.isdir(os.path.join(self.data_path, item)) and 'Rat' in item:
                # Add the directory to the dictionary
                subfolders[item] = os.path.join(self.data_path, item)

        # Iterate every rat to create the structure do save the shanks for each side
        for rat,path in subfolders.items():
            # Create the rat dict
            maps[rat] = {}
            # Walk through the directory
            for dirpath, dirnames, _ in os.walk(path):
                # If the current directory does not contain any subdirectories
                if not dirnames:
                    # Add the session name to the dictionary as key and create the left and right shanks
                    session_name = dirpath.split(os.sep)[-1]
                    maps[rat][session_name] = {'left':[],'right':[]}

        # Create the reference for rat name as key and rat number as value
        rats_dict = {'Rat08':8,'Rat09':9,'Rat10':10,'Rat11':11}
        # Create the reference dict of left and right shanks threshold (based on maps figure)
        side = {'Rat08':12,'Rat09':8,'Rat10':8,'Rat11':8}

        # Iterate every rat
        for rat in maps:
            # Iterate each session
            for idx,session in enumerate(maps[rat]):
                # If is rat 8 or 9 use the 'less equal' for left and greater for right (based on maps figure)
                if rat == 'Rat08' or rat == 'Rat09':
                    left = [shank for shank in self.structure[(self.structure[:,0]==rats_dict[rat]) & 
                                                            (self.structure[:,1]==idx+1)][:,2] 
                                                            if np.less_equal(shank,side[rat])]
                    right = [shank for shank in self.structure[(self.structure[:,0]==rats_dict[rat]) & 
                                                             (self.structure[:,1]==idx+1)][:,2]
                                                             if np.greater(shank,side[rat])]
                    # Fill the dict
                    maps[rat][session]['left'] = left
                    maps[rat][session]['right'] = right
                elif rat == 'Rat10' or rat == 'Rat11':
                    # If is rat 10 or 11 use the 'less equal' for right and greater for left (based on maps figure)
                    left = [shank for shank in self.structure[(self.structure[:,0]==rats_dict[rat]) &
                                                            (self.structure[:,1]==idx+1)][:,2]
                                                            if np.greater(shank,side[rat])]
                    right = [shank for shank in self.structure[(self.structure[:,0]==rats_dict[rat]) & 
                                                             (self.structure[:,1]==idx+1)][:,2] 
                                                             if np.less_equal(shank,side[rat])]
                    # Fill the dict
                    maps[rat][session]['left'] = left
                    maps[rat][session]['right'] = right   

        # Iterate every rat
        for rat in list(maps.keys()):
            # Iterate each session
            for session in list(maps[rat].keys()):
                # Remove the sessions that doesn't have structure data
                if len(maps[rat][session]['left']) == 0 and len(maps[rat][session]['right']) == 0:
                    del maps[rat][session] 

        # Return the maps dictionary filtering by the rats choosed in the initialisation
        if self.rats is None:
            return maps
        else:
            maps = {rat: maps[rat] for rat in self.rats}
            return maps

    def _get_states(self):
        """
        Processes and combines data from multiple sources to generate a dictionary of state intervals.
        This function initializes an empty dictionary to store the state intervals. It then iterates over 
        each rat in the `maps` dictionary (to make sure to just loop in folders that have the desired info). 
        For each rat, it initializes an empty dictionary to store the states for each session. It then 
        iterates over each session for the current rat. For each session, it initializes an empty dictionary 
        to store the states. It loads the state data for the current rat and session. It gets the keys for 
        the state data, skipping the first three keys (usually metadata). For each state, it initializes an 
        empty dictionary to store the state intervals. It then extracts interval data from the raw state data, 
        calculates the start time, end time, and duration for each interval, and stores the start time, 
        end time, and duration in the dictionary. The code check if the file exists to load it, otherwise the
        specific session will be tagged with 'Missing File' in the dictionary.

        States: 'Rem': REM sleep, 'sws': NREM sleep, 'wake': Wake, 'drowsy': drowsy.

        Returns
        -------
        dict
            A dictionary that mapss rats to their sessions, and each session to the corresponding state intervals.

        Notes
        -----
        This function assumes that the `maps`, `paths`, and `t_ref` are attributes of the class.
        It's important to check the dataset folder to see if all files are there, this function
        was used mainly with BLA data.

        Examples
        --------
        >>> self.paths = {'Rat01': {'Session1': {'States': 'path/to/states'}}}
        >>> self.t_ref = 1000 # is using time_ref = 'ms'
        >>> self.get_states()
        {'Rat01':                            
                {'Session1': {'Rem': {0: [0.0, 1.0, 1.0], 1: [2.0, 3.0, 1.0]}, 
                              'sws': {0: [0.0, 1.0, 1.0], 1: [2.0, 3.0, 1.0]}}}}
        states['Rat01']['Session1']['Rem'] = [start, stop, duration]                     
                
        Author
        ------
        written by Tulio Almeida 11/2023
        """
        # Initialize an empty dictionary to store the states intervals
        states = {}

        # Iterate over unique rats in metadata
        for rat in self.rat_idx:
            # For each rat, initialize an empty dictionary to store the states for each session
            states[rat] = {}

            # Iterate over days for each rat
            for session in self.maps[rat]:
                # If the file exists, read and prepare the dicttonary
                if os.path.exists(self.paths[rat][session]['States']):
                    # For each session, initialize an empty dictionary to store the states
                    states[rat][session] = {}

                    # Load the state data for the current rat and session
                    states_raw = sio.loadmat(self.paths[rat][session]['States'])

                    # Get the keys for the state data, skipping the first three keys (usually metadata)
                    states_keys = list(states_raw.keys())[3:]

                    # For each state, initialize an empty dictionary to store the state intervals
                    for state in states_keys:
                        states[rat][session][state] = {}

                        # Extract interval data from raw state data
                        for idx, interval in enumerate(states_raw[state]):
                            # Calculate the start time, end time, and duration for each interval
                            start_time, end_time = interval[0] * self.t_ref, interval[1] * self.t_ref
                            duration = end_time - start_time

                            # Store the start time, end time, and duration in the dictionary
                            states[rat][session][state][idx] = [start_time, end_time, duration]
                
                # If the file doens't exists, track it
                else:
                    self.__warning_msg(self.paths[rat][session]['States'])
                    states[rat][session] = 'Missing File'

        # Return the states dictionary
        return states

    def _get_states_ordered(self):
        """
        Return a dictionary containing DataFrames ordered by 'Start' and 'End' of each sleep state
        for each rat and session.
        This function initializes an empty dictionary to store the states intervals. It then iterates 
        over unique rats in metadata and for each rat, initializes an empty dictionary to store the 
        states for each session. It then iterates over days for each rat and creates a DataFrame for 
        each session. It then iterates over states and concatenates states in a single DataFrame. 
        It then orders the DataFrame based on event start and end and stores the ordered DataFrame 
        in the dict.         

        Parameters
        ----------
        self : object
            The object instance.

        Returns
        -------
        dict
            A dictionary containing ordered DataFrames of states for each rat and session.
               
        Author
        ------
        written by Tulio Almeida 11/2023
        """
        # Initialize an empty dictionary to store the states intervals
        states = {}

        # Iterate over unique rats in metadata
        for rat in self.rat_idx:
            # For each rat, initialize an empty dictionary to store the states for each session
            states[rat] = {}

            # Iterate over days for each rat
            for session in self.maps[rat]:
                # Create the DataFrame for session 
                df = pd.DataFrame()

                # Iterate over states and concatenate states in a single DataFrame
                for state in self.states[rat][session]:
                    df_temp = pd.DataFrame(self.states[rat][session][state]).T
                    df_temp.rename(columns = {0:'Start',1:'End', 2:'Duration'}, inplace = True)
                    df_temp['State'] = [state] * len(df_temp)
                    df = pd.concat([df, df_temp], ignore_index = True)

                # Order the DataFrame based on event start and end
                df = df.sort_values(by=['Start', 'End'])
                # Store the ordered DataFrame in the dict
                states[rat][session] = df

        # Return the states dictionary
        return states

    def _get_pos(self):
        """
        This method initializes an empty dictionary to store the position data for each rat and session.

        This function initializes an empty dictionary to store the tracking data. It then iterates over each 
        rat in the `maps` dictionary ('filter' the load by the maps used). For each rat, it initializes 
        an empty dictionary to store the tracking for each session. It then iterates over each session 
        for the current rat. For each session, it initializes an empty dictionary to store the tracking
        in a DataFrame(with columns 'Time','x','y'). 
        It then identifies the tracking file in the current rat and session. If the file does not exist, it 
        stores 'Missing file' in the dictionary.

        Returns
        -------
        dict
            A dictionary where the keys are rats and the values are dictionaries. Each nested dictionary 
            has session keys and pandas DataFrame values representing the position data for each session.

        Examples
        --------
        >>> self._get_pos()
        {'Rat01': {'Session1': {DataFrame['Time','x','y']}}}

        Author
        -------
        written by Tulio Almeida 11/2023       
        """
        # Initialize an empty dictionary to store the states intervals
        pos = {}

        # Iterate over unique rats in metadata
        for rat in self.rat_idx:
            # For each rat, initialize an empty dictionary to store the tracking data for each session
            pos[rat] = {}

            # Iterate over days for each rat
            for session in self.maps[rat]:
                # If the file exists, read and prepare the dictionary
                if os.path.exists(self.paths[rat][session]['position']):
                    # Load the tracking data
                    df = pd.read_csv(self.paths[rat][session]['position'])
                    df.rename(columns={'Time (us)': 'Time'}, inplace=True)
                    # If time ref is in seconds
                    if self.t_ref == 1:
                        df['Time'] = df['Time'].values/(self.t_ref * 1e6)
                    # If time ref is in ms
                    elif self.t_ref == 1000:
                        df['Time'] = df['Time'].values/self.t_ref
                    
                    # For each session, initialize an empty dictionary to store the tracking data
                    pos[rat][session] = df

                # If the tracking data file does not exist, store 'Missing file' in the LFP dictionary
                else:
                    self.__warning_msg(self.paths[rat][session]['position'])
                    pos[rat][session] = 'Missing file'   
        
        # Return the tracking data in dataframes stored in a dictionary
        return pos
       
    def _get_events(self):
        """
        Processes and combines data from multiple sources to generate a dictionary of events.

        This function initializes an empty dictionary to store the events. It then iterates over each 
        rat in the `maps` dictionary ('filter' the load by the maps used). For each rat, it initializes 
        an empty dictionary to store the events for each session. It then iterates over each session 
        for the current rat. For each session, it initializes an empty dictionary to store the events. 
        It then identifies the event files in the current rat and session. For each event file, if the 
        file exists, it reads the event data into a pandas DataFrame. If the file does not exist, it 
        stores 'Missing file' in the dictionary. Each event is stored with a specific key and as a
        DataFrame with the colums time stamp and data (DataFrame[ts,data]).

        Returns
        -------
        dict
            A dictionary that mapss rats to their sessions, and each session to the corresponding events.

        Examples
        --------
        >>> self._get_event()
        {'Rat01': {'Session1': {'event1': DataFrame([ts,data]), 'event2': DataFrame([ts,data])}}}

        Author
        ------
        written by Tulio Almeida 11/2023
        """
        # Initialize an empty dictionary to store the events
        events = {}

        # Iterate over each rat
        for rat in self.rat_idx:
            # For each rat, initialize an empty dictionary to store the events for each session
            events[rat] = {}

            # Iterate over each session for the current rat
            for session in self.maps[rat]:
                # For each session, initialize an empty dictionary to store the events
                events[rat][session] = {}

                # Identify the event files in the current rat and session
                files = {key:path for key,path in self.paths[rat][session].items() if str(path).endswith('.evt')}

                # Iterate over each event file
                for key,path in files.items():
                    # If the file exists, read the event data into a pandas DataFrame
                    if os.path.exists(path) and key == 'cat':
                        df =  pd.read_table(self.paths[rat][session][key],
                                            header = None,
                                            delimiter=' ')
                        data_values = list(df[1].values + ' ' + [file.split('-')[-1] for file in (df[3].values)])
                        df.columns = ['ts','0','1','data']
                        df['data'] = data_values
                        df = df.drop(columns=['0', '1'])
                        if self.t_ref == 1:
                            df['ts'] = df['ts'].values / self.t_ref
                        events[rat][session][key] = df
                    elif os.path.exists(path):
                        df =  pd.read_table(self.paths[rat][session][key],
                                            header = None)
                        df.columns = ['ts','data']
                        if key != 'rip' and self.t_ref == 1:
                            df['ts'] = df['ts'].values / self.t_ref
                        elif key == 'rip' and self.t_ref == 1000:
                            df['ts'] = df['ts'].values * self.t_ref
                        events[rat][session][key] = df

                    # If the file does not exist, store 'Missing file' in the dictionary
                    else:
                        self.__warning_msg(path)
                        events[rat][session][key] = 'Missing file'

        # Return the dictionary of events
        return events

    def get_lfp_ch(self):
        """
        This function will create a .xlsx for each rat. You must pass the channels for each session 
        in it (check README.md file for example and more details).
        This method initializes an empty dictionary to store the spike times. It then iterates 
        over each rat, and if a channel file for the rat does not exist in the Global_vars folder,
        it adds the rat to the dictionary and appends the session data to the rat's list in the 
        dictionary. Finally, it saves the dictionary to an Excel file. 

        Returns
        -------
        dict
            A dictionary where the keys are rat names and the values are lists of session data.

        Author
        ------
        written by Tulio Almeida 11/2023
        """
        # Initialize an empty dictionary to store the spike times
        lfp_dict = {}

        # Iterate over each rat
        for rat in self.rat_idx:
            # Check if the channel file for the rat already exists in the Global_vars folder
            if os.path.exists(os.path.abspath(os.path.join(self.data_path,
                                                           f"Global_vars\\{rat}_{self.struct}_ch.xlsx"))):
                # If the file exists, print a message
                print(f'The channel file for {rat} for {self.struct} already exist in Global_vars folder!')
            else:
                # If the file does not exist, add the rat to the dictionary
                lfp_dict[rat] = []

                # Iterate over each session for the rat
                for _, session in enumerate(self.maps[rat]):
                    # Append the session data to the rat's list in the dictionary
                    lfp_dict[rat].append(session)
                            
                # Define the path for the channel file
                df_path = os.path.abspath(os.path.join(self.data_path,
                                                       f"Global_vars\\{rat}_{self.struct}_ch.xlsx"))
                # Create a DataFrame from the dictionary
                df = pd.DataFrame({rat:lfp_dict[rat],'Channels':[None for i in range(len(lfp_dict[rat]))]}) 
                # Save the DataFrame to an Excel file
                df.to_excel(df_path, index=False) 

    def __get_channels(self, rat):
        """
        Get a dictionary of channels for a given rat.

        This method reads an Excel file that contains the channels for a given rat, and returns 
        a dictionary where the keys are session names and the values are either a single channel 
        or a list of channels.

        Parameters
        ----------
        rat : str
            The name of the rat.

        Returns
        -------
        dict
            A dictionary where the keys are session names and the values are either a single 
            channel or a list of channels.

        Author
        ------
        written by Tulio Almeida 12/2023
        """
        # Check if the file for the channels for the structure already exist
        path_xlsx = os.path.abspath(os.path.join(self.data_path,f"Global_vars\\{rat}_{self.struct}_ch.xlsx"))
        # If not return an error message
        assert os.path.exists(path_xlsx) is True, (self.get_lfp_ch(),f'Fill the file: {path_xlsx}')
        # Load the xlsx file with the channels info
        df = pd.read_excel(path_xlsx)
        # Check if there are any session without an channel
        soma = sum([1 if np.isnan(ch) else 0 for ch in df['Channels']])
        # If there are channel without a channel rise a error
        assert soma == 0,f'Add at least one channel in all sessions in the file: {path_xlsx}'
        # Initialize the empty dict to retrive session channel for each rat
        ch_dict = {}

        # Iterate over session/channel in the DataFrame
        for session,ch in df.values:
            # If is a single channel store it in directly
            if type(ch) == int:
                ch_dict[session] = ch

            # Otherwise wplit each channel in the cell and store it
            else:
                ch_dict[session] = [int(channel) for channel in ch.split(',')]

        # Return the dict with session as key and the correspondent channels
        return ch_dict

    def __save_hdf5(self, data, name):
        """
        Save a dictionary to an hdf5 file.
        This method saves a dictionary to an hdf5 file. The dictionary can contain other dictionaries
        (which will be saved recursively) and numpy arrays (which will be saved directly). If the dictionary
        contains any other types of items, an error will be raised.

        Parameters
        ----------
        data : dict
            The dictionary to save to the hdf5 file.
        name : str
            The name of the hdf5 file.

        Raises
        ------
        ValueError
            If the dictionary contains an item that is neither a dictionary nor a numpy array.

        Examples
        --------
        >>> # Create a dictionary with a numpy array and a nested dictionary
        >>> data = {'array': np.array([1, 2, 3]), 'nested': {'key': 'value'}}
        >>> # Save the dictionary to an hdf5 file
        >>> __save_hdf5(data, 'Rat08_Hpc')

        Author
        ------
        written by Tulio Almeida 12/2023
        """
        # Define the absolute path to the hdf5 file
        path = os.path.abspath(self.data_path + os.sep + 'Global_vars' + os.sep + name)

        # Open the hdf5 file in write mode
        with h5py.File(path, 'w') as h5file:
            # Define a function to save a dictionary to the hdf5 file
            def save_dict_to_hdf5(h5file, path, dic):
                # Iterate over each item in the dictionary
                for key, item in dic.items():
                    # If the item is a dictionary, call the function recursively
                    if isinstance(item, dict):
                        save_dict_to_hdf5(h5file, path + key + os.sep, item)

                    # If the item is a numpy array, save it to the hdf5 file
                    elif isinstance(item, np.ndarray):
                        h5file[path + str(key)] = item

                    # If the item is neither a dictionary nor a numpy array, raise an error
                    else:
                        raise ValueError('Cannot save %s type' % type(item))

            # Call the function to save the data dictionary to the hdf5 file
            save_dict_to_hdf5(h5file, os.sep, data)

    def __load_hdf5(self,h5file, path):
        """
        Load data from an HDF5 file into a dictionary.

        This method iterates over each key in the HDF5 file at the given path.
        If the item at the current key is a group (i.e., a nested dictionary), the method
        calls itself recursively. If the item is a dataset (i.e., a numpy array), the method
        gets the data from the dataset.

        Parameters
        ----------
        h5file : h5py.File
            The HDF5 file from which to load data.
        path : str
            The path in the HDF5 file from which to load data.

        Returns
        -------
        dict
            A dictionary containing the data loaded from the HDF5 file.

        Examples
        --------
        >>> h5file = h5py.File('test.hdf5', 'r')
        >>> path = '/path/to/data'
        >>> my_object = MyClass()
        >>> data = my_object.__load_hdf5(h5file, path)
        >>> print(data)
        {'key1': {'key2': array([1, 2, 3])}, 'key3': array([4, 5, 6])}

        Author
        ------
        written by Tulio Almeida 12/2023
        """
        # Initialize an empty dictionary
        dic = {}

        # Iterate over each key in the hdf5 file at the given path
        for key in h5file[path].keys():
            # Get the item at the current key in the hdf5 file
            item = h5file[path + key]

            # If the item is a group (i.e., a nested dictionary), call the __load_hdf5 method recursively
            if isinstance(item, h5py.Group):
                dic[key] = self.__load_hdf5(h5file, path + key + '/')

            # If the item is a dataset (i.e., a numpy array), get the data from the dataset
            elif isinstance(item, h5py.Dataset):
                dic[key] = item[()]

        # Return the dictionary
        return dic

    def load_lfp(self,path):
        """
        Load local field potential (LFP) data from an HDF5 file. This function 
        is used inside the 'get_lfp()' but can be used outside of it. It opens an HDF5 file, 
        loads the data into a dictionary, and then organizes the data into a nested dictionary. 
        The keys of the nested dictionary are derived from the original dictionary keys.

        Parameters
        ----------
        path : str
            The path to the HDF5 file.

        Returns
        -------
        lfp : dict
            A dictionary containing the LFP data.

        Notes
        -----
        The function uses the h5py library to handle HDF5 files and the
        collections library to create a nested dictionary.

        Examples
        --------
        >>> instance.load_lfp('path_to_hdf5_file')

        Author
        ------
        written by Tulio Almeida 12/2023
        """
        # If it does, open the HDF5 file in read mode
        with h5py.File(path, 'r') as h5file:
            # Load the HDF5 file into a dictionary
            lfp_dict = self.__load_hdf5(h5file, '/')

        # Initialize a nested dictionary to store the LFP data
        nested_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        # Iterate over each item in the LFP dictionary
        for key, value in lfp_dict.items():
            # Split the key into its components
            keys = key.split(os.sep)
            # Store the value in the nested dictionary using the components of the key as indices
            if keys[3].isdigit():
                nested_dict[keys[1]][keys[2]][int(keys[3])] = value
            else:
                nested_dict[keys[1]][keys[2]][keys[3]] = value

        # Convert the nested dictionary into a regular dictionary
        lfp = dict(nested_dict)

        # Return the LFP data
        return lfp

    def get_lfp(self):
        """
        Load local field potential (LFP) data and save it in an HDF5 file.

        This function checks if the LFP file for a specific structure already exists.
        If it does, it opens the HDF5 file in read mode, loads the HDF5 file into a dictionary,
        and then restructures the dictionary into a nested dictionary.

        If the LFP file does not exist, it initializes an empty dictionary to store the LFPs.
        It then iterates over each rat, initializes an empty dictionary to store the events for each session,
        and reads the event data into a pandas DataFrame. If the file does not exist, it stores 'Missing file'
        in the dictionary.

        If the save attribute is True, it saves the LFP data in an HDF5 file inside Global_vars folder.

        Notes
        -----
        The function uses the h5py library to handle HDF5 files and the
        collections library to create a nested dictionary.
        
        Returns
        -------
        dict
            A dictionary containing the LFP data.

        Examples
        --------
        >>> lfp_data = self.get_lfp()

        Author
        ------
        written by Tulio Almeida 12/2023
        """
        # Define the path to the LFP file for the current structure
        name = self.struct + '_lfp.hdf5'
        lfp_path = os.path.abspath(self.data_path + os.sep + 'Global_vars' + os.sep + name)

        # Check if the LFP file already exists
        if os.path.exists(lfp_path):
            return self.load_lfp(lfp_path)

        else:
            # If the LFP file does not exist, initialize an empty dictionary to store the LFP data
            lfp = {}

            # Iterate over each rat
            for rat in self.rat_idx:
                # For each rat, initialize an empty dictionary to store the events for each session
                lfp[rat] = {}
                # Get the channels for the current rat
                ch_dict = self.__get_channels(rat)
                # Iterate over each session for the current rat
                for session,_ in ch_dict.items():
                    # For each session, initialize an empty dictionary to store the events
                    lfp[rat][session] = {}
                    # Get the path to the LFP data for the current rat and session
                    path = self.paths[rat][session]['lfp']
                    
                    # Check if the LFP data file exists
                    if os.path.exists(path):
                        # If it does, read the LFP data into a pandas DataFrame
                        lfp_data = neo.NeuroScopeIO(path)
                        block = lfp_data.read_block(signal_group_mode='split-all',lazy=True)
                        del lfp_data
                        # If the channels are integers, load the data for each channel and store it in the LFP dictionary
                        if type(ch_dict[session]) == int:
                            data = block.segments[0].analogsignals[ch_dict[session]].load()
                            lfp[rat][session][ch_dict[session]] = np.array(data, dtype=np.float64).flatten()
                            del data
                            # Create a time vector for the LFP data
                            lfp[rat][session]['Time'] = np.arange(0, len(lfp[rat][session][ch_dict[session]]),
                                                                dtype=np.float64)/(self.fs * self.t_ref)
                        else:
                            # If the channels are not integers, load the data for each channel and store it in the LFP dictionary
                            for channel in ch_dict[session]:
                                data = block.segments[0].analogsignals[channel].load()
                                lfp[rat][session][channel] = np.array(data, dtype=np.float64).flatten()
                                del data

                            # Create a time vector for the LFP data
                            lfp[rat][session]['Time'] = np.arange(0, len(lfp[rat][session][channel]), 
                                                                dtype=np.float64)/(self.fs * self.t_ref)
                        del block

                    # If the LFP data file does not exist, store 'Missing file' in the LFP dictionary
                    else:
                        self.__warning_msg(path)
                        lfp[rat][session][0] = 'Missing file'
            
            # If the save attribute is True, save the LFP data in an HDF5 file
            if self.save:
                self.__save_hdf5(lfp, name)

            # Return the LFP data
            return lfp

    def load_spk(self,path):
        """
        Load spike data from an HDF5 file. This function is used inside the 'get_spk()' but can be used outside 
        of it, for example to load spike data from pyramidal cells and interneurons. It opens an HDF5 file, 
        loads the data into a dictionary, and then organizes the data into a nested dictionary. 
        The keys of the nested dictionary are derived from the original dictionary keys.

        Parameters
        ----------
        path : str
            The path to the HDF5 file.

        Returns
        -------
        spk : dict
            A dictionary containing the spike data.

        Notes
        -----
        The function uses the h5py library to handle HDF5 files and the
        collections library to create a nested dictionary.

        Examples
        --------
        >>> instance.load_spk('path_to_hdf5_file')

        Author
        ------
        written by Tulio Almeida 12/2023
        """
        # If it does, open the HDF5 file in read mode
        with h5py.File(path, 'r') as h5file:
            # Load the HDF5 file into a dictionary
            spk_dict = self.__load_hdf5(h5file, '/')

        # Initialize a nested dictionary to store the spike data
        nested_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        # Iterate over each item in the spike dictionary
        for key, value in spk_dict.items():
            # Split the key into its components
            keys = key.split(os.sep)
            # Store the value in the nested dictionary using the components of the key as indices
            nested_dict[keys[1]][keys[2]][keys[3]][int(keys[4])] = value

        # Convert the nested dictionary into a regular dictionary
        spk = dict(nested_dict)

        # Return the spike data
        return spk
        
    def get_spk(self):
        """
        Load spike data and save it in an HDF5 file.

        This function checks if the spike file for a specific structure already exists.
        If it does, it opens the HDF5 file in read mode, loads the HDF5 file into a dictionary,
        and then restructures the dictionary into a nested dictionary.

        If the spike file does not exist, it initializes an empty dictionary to store the spikes.
        It then iterates over each rat, initializes an empty dictionary to store the events for each session,
        and reads the event data into a pandas DataFrame. If the file does not exist, it stores 'Missing file'
        in the dictionary.

        If the save attribute is True, it saves the LFP data in an HDF5 file inside Global_vars folder.

        The output will be a dictiory with rat/session/spikes, spikes is a dict that contains data from
        spikes in the 'left' and 'right' structure, and merged in 'all' key (keys = ['all', 'left', 'right']).
        Each spike have a unique ID extracted from the Original Dataset, this ID was developed by Gabrielle
        Girardeau and it's important to compare the results. 

        Notes
        -----
        The function uses the h5py library to handle HDF5 files and the
        collections library to create a nested dictionary.

        Returns
        -------
        dict
            A dictionary that mapss rats to their sessions, and each session to the corresponding spike times.

        Notes
        -----
        The keys in the spike output are related to the unique ID developed by Gabrielle Girardeau and used in
        the Original Dataset.

        Examples
        --------
        >>> self.get_spk()
        {'Rat01': {'Session1': {'left': {1: array([1, 2]), 2: array([3, 4])}, 'right': {3: array([5, 6]), 
        4: array([7, 8])}, 'all': {1: array([1, 2]), 2: array([3, 4]), 3: array([5, 6]), 4: array([7, 8])}}}

        Author
        -------
        written by Tulio Almeida 11/2023
        """
        # Define the path to the LFP file for the current structure
        if self.cell == 1:
            name = self.struct + '_spkPyr.hdf5'
        elif self.cell == 2:
            name = self.struct + '_spkInt.hdf5'
        
        spk_path = os.path.abspath(self.data_path + os.sep + 'Global_vars' + os.sep  + name)

        # Check if the spike file already exists, if yes load the data
        if os.path.exists(spk_path): 
            return self.load_spk(spk_path)
        
        else:
            # Initialize an empty dictionary to store the spike times
            spk = {}

            # Iterate over each rat
            for rat in self.rat_idx:
                # For each rat, initialize an empty dictionary to store the spike times for each session
                spk[rat] = {}

                # Iterate over each session for the current rat
                for _, session in enumerate(self.maps[rat]):
                    # For each session, initialize an empty dictionary to store the spike times 
                    # for the left and right shanks
                    spk[rat][session] = {'left': {}, 'right': {}}

                    # Load the index data for the current rat and session
                    idx_temp = sio.loadmat(self.paths[rat][session]['index'])['Index']

                    # Load the spike data for the current rat and session
                    spk_temp = sio.loadmat(self.paths[rat][session]['spike'])['spikes']

                    # Filter the cells for the left shank based on the rat, day, and cluster
                    cell_l = self.final_type[(self.final_type[:,0] == self.rat_idx[rat]) & 
                                            (self.final_type[:,1] == self.paths[rat][session]['day']) &
                                            (np.isin(self.final_type[:,2], self.maps[rat][session]['left']))] 

                    # Filter the cells for the right shank based on the rat, day, and cluster
                    cell_r = self.final_type[(self.final_type[:,0] == self.rat_idx[rat]) &
                                            (self.final_type[:,1] == self.paths[rat][session]['day']) &
                                            (np.isin(self.final_type[:,2], self.maps[rat][session]['right']))] 

                    # Identify the clusters of cells for the left shank
                    clusters_l = set(idx_temp[np.isin(idx_temp[:,2], self.maps[rat][session]['left'])][:,3])

                    # Identify the clusters of cells for the right shank
                    clusters_r = set(idx_temp[np.isin(idx_temp[:,2], self.maps[rat][session]['right'])][:,3])

                    # If there are any clusters for the left shank, iterate over each shank and cluster
                    if len(clusters_l) > 0:
                        for shank in self.maps[rat][session]['left']:
                            for cluster in clusters_l:
                                # For each cell that matches the specified cell type, add the spike times to the dict
                                id_ref = idx_temp[(idx_temp[:,2]==shank) & (idx_temp[:,3]==cluster)]
                                cell_ref = cell_l[(cell_l[:,2]==shank) & (cell_l[:,3]==cluster)]
                                if len(id_ref) > 0 and cell_ref[0,4] == self.cell: 
                                    id = id_ref[0,4]
                                    ts = spk_temp[(spk_temp[:,1]== shank) & (spk_temp[:,2]== cluster)][:,0]
                                    spk[rat][session]['left'][id] = ts * self.t_ref

                    # If there are any clusters for the right shank, iterate over each shank and cluster
                    if len(clusters_r) > 0:
                        for shank in self.maps[rat][session]['right']:
                            for cluster in clusters_r:
                                # For each cell that matches the specified cell type, add the spike times to the dict
                                id_ref = idx_temp[(idx_temp[:,2]==shank) & (idx_temp[:,3]==cluster)]
                                cell_ref = cell_r[(cell_r[:,2]==shank) & (cell_r[:,3]==cluster)]
                                if len(id_ref) > 0 and cell_ref[0,4] == self.cell: 
                                    id = id_ref[0,4]
                                    ts = spk_temp[(spk_temp[:,1]== shank) & (spk_temp[:,2]== cluster)][:,0]
                                    spk[rat][session]['right'][id] = ts * self.t_ref

                    # Combine the left and right spike times into a single dictionary for each rat and session
                    if rat == 'Rat08' or rat == 'Rat09':
                        spk[rat][session]['all'] = {**spk[rat][session]['left'], **spk[rat][session]['right']}
                    elif rat == 'Rat10' or rat == 'Rat11': 
                        spk[rat][session]['all'] = {**spk[rat][session]['right'], **spk[rat][session]['left']}
            
            # If the save attribute is True, save the spike data in an HDF5 file
            if self.save:
                self.__save_hdf5(spk, name)

            # Return the spikes dictionary
            return spk