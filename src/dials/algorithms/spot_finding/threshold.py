from __future__ import annotations


class ThresholdStrategy:
    """
    Base class for spot finder threshold strategies.
    """

    def __init__(self, **kwargs):
        """
        Initialise with key word arguments.
        """
        pass

    def __call__(self, image):
        """
        Threshold the image.
        """
        raise RuntimeError("Overload Me!")


class DispersionThresholdStrategy(ThresholdStrategy):
    """
    A class implementing a 'gain' threshold.
    """

    def __init__(self, **kwargs):
        """
        Set the threshold algorithm up
        """

        # Initialise the base class
        ThresholdStrategy.__init__(self, **kwargs)

        # Get the parameters
        self._kernel_size = kwargs.get("kernel_size", (3, 3))
        self._gain = kwargs.get("gain")
        self._n_sigma_b = kwargs.get("n_sigma_b", 6)
        self._n_sigma_s = kwargs.get("n_sigma_s", 3)
        self._min_count = kwargs.get("min_count", 2)
        self._threshold = kwargs.get("global_threshold", 0)

        # Save the constant gain
        self._gain_map = None

    def __call__(self, image, mask, algorithm_cache=None):
        """
        Call the thresholding function

        :param image: The image to process
        :param mask: The mask to use
        :param algorithm_cache: A dict in which to keep the thresholder, so that
                                images of the same shape reuse it. It holds a
                                summed-area table per shape, so its lifetime bounds
                                that of the table; a private one is used if none is
                                given.
        :return: The thresholded image
        """
        from dials.algorithms.image import threshold
        from dials.array_family import flex

        # Initialise the algorithm
        if algorithm_cache is None:
            algorithm_cache = {}
        try:
            algorithm = algorithm_cache[image.all()]
        except KeyError:
            algorithm = threshold.DispersionThreshold(
                image.all(),
                self._kernel_size,
                self._n_sigma_b,
                self._n_sigma_s,
                self._threshold,
                self._min_count,
            )
            algorithm_cache[image.all()] = algorithm

        # Set the gain
        if self._gain is not None:
            assert self._gain > 0
            if self._gain_map is None or self._gain_map.all() != image.all():
                self._gain_map = flex.double(image.accessor(), self._gain)

        # Compute the threshold
        result = flex.bool(flex.grid(image.all()))
        if self._gain_map:
            algorithm(image, mask, self._gain_map, result)
        else:
            algorithm(image, mask, result)

        # Return the result
        return result


class DispersionExtendedThresholdStrategy(ThresholdStrategy):
    """
    A class implementing a 'gain' threshold.
    """

    def __init__(self, **kwargs):
        """
        Set the threshold algorithm up
        """

        # Initialise the base class
        ThresholdStrategy.__init__(self, **kwargs)

        # Get the parameters
        self._kernel_size = kwargs.get("kernel_size", (3, 3))
        self._gain = kwargs.get("gain")
        self._n_sigma_b = kwargs.get("n_sigma_b", 6)
        self._n_sigma_s = kwargs.get("n_sigma_s", 3)
        self._min_count = kwargs.get("min_count", 2)
        self._threshold = kwargs.get("global_threshold", 0)

        # Save the constant gain
        self._gain_map = None

    def __call__(self, image, mask, algorithm_cache=None):
        """
        Call the thresholding function

        :param image: The image to process
        :param mask: The mask to use
        :param algorithm_cache: A dict in which to keep the thresholder, so that
                                images of the same shape reuse it. It holds a
                                summed-area table per shape, so its lifetime bounds
                                that of the table; a private one is used if none is
                                given.
        :return: The thresholded image
        """
        from dials.algorithms.image import threshold
        from dials.array_family import flex

        # Initialise the algorithm
        if algorithm_cache is None:
            algorithm_cache = {}
        try:
            algorithm = algorithm_cache[image.all()]
        except KeyError:
            algorithm = threshold.DispersionExtendedThreshold(
                image.all(),
                self._kernel_size,
                self._n_sigma_b,
                self._n_sigma_s,
                self._threshold,
                self._min_count,
            )
            algorithm_cache[image.all()] = algorithm

        # Set the gain
        if self._gain is not None:
            assert self._gain > 0
            if self._gain_map is None or self._gain_map.all() != image.all():
                self._gain_map = flex.double(image.accessor(), self._gain)

        # Compute the threshold
        result = flex.bool(flex.grid(image.all()))
        if self._gain_map:
            algorithm(image, mask, self._gain_map, result)
        else:
            algorithm(image, mask, result)

        # Return the result
        return result
