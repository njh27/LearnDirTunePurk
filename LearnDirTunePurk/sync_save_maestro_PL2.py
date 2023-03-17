import sys
import os
import pickle
from ReadMaestro.utils.PL2_maestro import NoEventsError
from LearnDirTunePurk.build_session import create_behavior_session, add_neuron_trials



if __name__ == '__main__':
    """
    """
    # python sync_save_maestro_PL2.py /home/nate/neuron_viz_final/

    # Neurons dir is only input and used to load neurons files and filenames
    neurons_dir = sys.argv[1]
    # These remaining dirs are HARD CODED
    maestro_dir = '/mnt/isilon/home/nathan/Data/LearnDirTunePurk/MaestroFiles/'
    PL2_dir = "/mnt/isilon/home/nathan/Data/LearnDirTunePurk/PL2FilesRaw/"
    skip = True
    start_fnum = 47
    for f in os.listdir(neurons_dir):
        fname = f
        fname = fname.split(".")[0]
        if fname[-4:].lower() == "_viz":
            fname = fname.split("_viz")[0]
        if fname[0:8].lower() == "neurons_":
            fname = fname.split("neurons_")[1]
        fnum = int(fname[-2:])
        if fnum == start_fnum:
            skip = False
        if skip:
            continue
        save_name = '/home/nate/ExpanDrive/OneDrive Business/Sync/LearnDirTunePurk/Data/Maestro/Pickles/' + fname + "_maestro"
        sess = create_behavior_session(fname, maestro_dir, session_name=fname, rotate=True,
                                                 check_existing_maestro=True,
                                                 save_maestro_data=True, save_maestro_name=save_name)


        fname_PL2 = fname + ".pl2"
        fname_neurons = "neurons_" + fname + "_viz.pkl"
        neurons_file = neurons_dir + fname_neurons
        print("Loading neurons from file {0}.".format(fname_neurons))
        with open(neurons_file, 'rb') as fp:
            neurons = pickle.load(fp)

        try:
            sess = add_neuron_trials(sess, maestro_dir, neurons_file, PL2_dir=PL2_dir,
                                               dt_data=1, save_maestro_name=save_name, save_maestro_data=True)
        except NoEventsError:
            print("!SKIPPING! file {0} because it has no PL2 events.".format(fname_PL2))
            continue
