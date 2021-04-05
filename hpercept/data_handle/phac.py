import matplotlib.pyplot as plt
from . import utils
class PHAC2:
    """

    General PHAC-2 class to represent the different attibutes of the data

    name: The descriptive name of the material
    accelerometer: Accelerometer reading
    electrode_0: Electrode readings in the BioTac finger 1
    pac_0: Dynamic pressure reading in the BioTac finger 1
    pdc_0: Static pressure reading in the BioTac finger 1
    tac_0: Heat flux reading in the BioTac finger 1
    tdc_0: Temperature reading in the BioTac finger 1
    electrode_1: Electrode readings in the BioTac finger 2
    pac_1: Dynamic pressure reading in the BioTac finger 2
    pdc_1: Static pressure reading in the BioTac finger 2
    tac_1: Heat flux reading in the BioTac finger 2
    tdc_1: Temperature reading in the BioTac finger 2
    controller_detail_state = The detailed state of the robotic controller
    image: The image representation of the data consisting from the composition of all the signals
    adjective: Adjectives defined by the human subjects
    """

    def __init__(self,
                 name=None,
                 accelerometer=None,
                 electrode_0=None,
                 pac_0=None,
                 pdc_0=None,
                 tac_0=None,
                 tdc_0=None,
                 electrode_1=None,
                 pac_1=None,
                 pdc_1=None,
                 tac_1=None,
                 tdc_1=None,
                 controller_detail_state=None,
                 image=None,
                 adjective=None):

        self.name = name
        self.accelerometer = accelerometer
        self.electrode_0 = electrode_0
        self.pac_0 = pac_0
        self.pdc_0 = pdc_0
        self.tac_0 = tac_0
        self.tdc_0 = tdc_0
        self.electrode_1 = electrode_1
        self.pac_1 = pac_1
        self.pdc_1 = pdc_1
        self.tac_1 = tac_1
        self.tdc_1 = tdc_1
        self.controller_detail_state = controller_detail_state
        self.image = image
        self.adjective = adjective

    def visualize(self,fn):
        """



        """

        fig, ax = plt.subplots(3, 4)

        fig.set_dpi(300)
        fig.set_figwidth(50)
        fig.set_figheight(20)

        ax[0][0].plot(self.accelerometer)
        ax[0][0].set_title("Accelerometer (BioTac 1)")
        ax[0][0].legend( ["$x$", "$y$", "$z$"], loc = "best")
        utils.specify_descriptive_boxes(self, ax[0][0])

        ax[0][1].plot(self.electrode_0)
        ax[0][1].set_title("Electrode (BioTac 1)")
        utils.specify_descriptive_boxes(self, ax[0][1])

        ax[1][0].plot(self.pac_0)
        ax[1][0].set_title("Dynamic Pressure (BioTac 1)")
        utils.specify_descriptive_boxes(self, ax[1][0])

        ax[1][1].plot(self.pdc_0)
        ax[1][1].set_title("Static Pressure (BioTac 1)")
        utils.specify_descriptive_boxes(self, ax[1][1])

        ax[2][0].plot(self.tac_0)
        ax[2][0].set_title("Heat Flux (BioTac 1)")
        utils.specify_descriptive_boxes(self, ax[2][0])

        ax[2][1].plot(self.tdc_0)
        ax[2][1].set_title("Temperature (BioTac 1)")
        utils.specify_descriptive_boxes(self, ax[2][1])

        ax[0][2].imshow(self.image)
        #ax[0][2].set_title("Accelerometer (BioTac 1)")

        ax[0][3].plot(self.electrode_1)
        ax[0][3].set_title("Electrode (BioTac 2)")
        utils.specify_descriptive_boxes(self, ax[0][3])

        ax[1][2].plot(self.pac_1)
        ax[1][2].set_title("Dynamic Pressure (BioTac 2)")
        utils.specify_descriptive_boxes(self, ax[1][2])

        ax[1][3].plot(self.pdc_1)
        ax[1][3].set_title("Static Pressure (BioTac 2)")
        utils.specify_descriptive_boxes(self, ax[1][3])

        ax[2][2].plot(self.tac_1)
        ax[2][2].set_title("Heat Flux (BioTac 2)")
        utils.specify_descriptive_boxes(self, ax[2][2])

        ax[2][3].plot(self.tdc_1)
        ax[2][3].set_title("Temperature (BioTac 2)")
        utils.specify_descriptive_boxes(self, ax[2][3])


        fig.suptitle("PHAC-2 Measurements of {}\nAdjectives = {}".format(self.name, "-".join(self.adjective)))
        fig.tight_layout()
        fig.savefig(fn)
        plt.close("all")

