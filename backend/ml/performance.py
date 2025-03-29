class PerformanceMetrics:
    @staticmethod
    def precision(tp : int, fp : int, tn : int, fn : int) -> float:
        '''
        Returns the precision given the number of true positives, false positives, true
        negatives, and false negatives.

        Parameters:
        -----------
        tp (int): the number of true positives
        fp (int): the number of false positives
        tn (int): the number of true negatives
        fn (int): the number of false negatives

        Returns:
        --------
        float: the precision
        '''
        return tp / (tp + fp)

    @staticmethod
    def recall(tp : int, fp : int, tn : int, fn : int) -> float:
        '''
        Returns the recall given the number of true positives, false positives, true
        negatives, and false negatives.

        Parameters:
        -----------
        tp (int): the number of true positives
        fp (int): the number of false positives
        tn (int): the number of true negatives
        fn (int): the number of false negatives

        Returns:
        --------
        float: the recall
        '''
        return tp / (tp + fn)

    @staticmethod
    def f1(tp : int, fp : int, tn : int, fn : int) -> float:
        '''
        Returns the F1-score given the number of true positives, false positives, true
        negatives, and false negatives.

        Parameters:
        -----------
        tp (int): the number of true positives
        fp (int): the number of false positives
        tn (int): the number of true negatives
        fn (int): the number of false negatives

        Returns:
        --------
        float: the F1-score
        '''
        precision = PerformanceMetrics.precision(tp, fp, tn, fn)
        recall = PerformanceMetrics.recall(tp, fp, tn, fn)
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def accuracy(tp : int, fp : int, tn : int, fn : int) -> float:
        '''
        Returns the accuracy given the number of true positives, false positives, true
        negatives, and false negatives.

        Parameters:
        -----------
        tp (int): the number of true positives
        fp (int): the number of false positives
        tn (int): the number of true negatives
        fn (int): the number of false negatives

        Returns:
        --------
        float: the accuracy
        '''
        return (tp + tn) / (tp + fp + tn + fn)
    
    @staticmethod
    def all_metrics(tp : int, fp : int, tn : int, fn : int) -> float:
        '''
        Returns the four performance given the number of true positives, false positives,
        true negatives, and false negatives.

        Parameters:
        -----------
        tp (int): the number of true positives
        fp (int): the number of false positives
        tn (int): the number of true negatives
        fn (int): the number of false negatives

        Returns:
        --------
        float: the accuracy
        float: the precision
        float: the recall
        float: the F1-score
        '''
        accuracy = PerformanceMetrics.accuracy(tp, fp, tn, fn)
        precision = PerformanceMetrics.precision(tp, fp, tn, fn)
        recall = PerformanceMetrics.recall(tp, fp, tn, fn)
        f1 = PerformanceMetrics.f1(tp, fp, tn, fn)
        return accuracy, precision, recall, f1

    