class PHAC2:
    """

    General PHAC-2 class to represent the different attibutes of the data

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
