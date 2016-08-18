import numpy as np
import numpy.linalg as la
import helpers

class Data(object):
    def __init__(self, opt):
        self.opt = opt
        self.measurements = {}
        self.num_loaded_samples = 0    # no of samples from file
        self.num_used_samples = 0      # no of samples after skipping
        self.samples = {}

        self.usedBlocks = list()
        self.unusedBlocks = list()
        self.seenBlocks = list()

        # has some data been loaded?
        self.inited = False

    def init_from_data(self, data):
        '''load data from numpy array'''

        self.samples = self.measurements = data
        self.num_loaded_samples = self.samples['positions'].shape[0]
        self.num_used_samples = self.num_loaded_samples/(self.opt['skip_samples']+1)
        if self.opt['verbose']:
            print 'loaded {} data samples (using {})'.format(
                self.num_loaded_samples, self.num_used_samples)
        self.inited = True

    def init_from_files(self, measurements_files):
        '''load data from measurements_files, optionally skipping some values'''

        with helpers.Timer() as t:
            # load data from multiple files and concatenate, fix timing
            for fa in measurements_files:
                for fn in fa:
                    m = np.load(fn)
                    mv = {}
                    for k in m.keys():
                        mv[k] = m[k]
                        if not self.measurements.has_key(k):
                            # first file
                            if m[k].ndim == 0:
                                self.measurements[k] = m[k]
                            elif m[k].ndim == 1:
                                self.measurements[k] = m[k][self.opt['start_offset']:]
                            else:
                                self.measurements[k] = m[k][self.opt['start_offset']:, :]
                        else:
                            # following files, append data
                            if m[k].ndim == 0:
                                #TODO: get mean value of scalar values (needs to count how many values then)
                                self.measurements[k] = m[k]
                            elif m[k].ndim == 1:
                                # let values start with first time diff
                                mv[k] = m[k] - m[k][0] + (m[k][1]-m[k][0])
                                mv[k] = mv[k] + self.measurements[k][-1]   # add after previous times
                                self.measurements[k] = np.concatenate( (self.measurements[k],
                                                                        mv[k][self.opt['start_offset']:]),
                                                                        axis=0)
                            else:
                                self.measurements[k] = np.concatenate( (self.measurements[k],
                                                                        mv[k][self.opt['start_offset']:, :]),
                                                                        axis=0)
                    m.close()

            self.num_loaded_samples = self.measurements['positions'].shape[0]
            self.num_used_samples = self.num_loaded_samples/(self.opt['skip_samples']+1)
            if self.opt['verbose']:
                print 'loaded {} measurement samples (using {})'.format(
                    self.num_loaded_samples, self.num_used_samples)

            # create data that identification is working on (subset of all measurements)
            self.samples = {}
            self.block_pos = 0
            if self.opt['selectBlocksFromMeasurements']:
                # fill with starting block
                for k in self.measurements.keys():
                    if self.measurements[k].ndim == 0:
                        self.samples[k] = self.measurements[k]
                    elif self.measurements[k].ndim == 1:
                        self.samples[k] = self.measurements[k][self.block_pos:self.block_pos + self.opt['block_size']]
                    else:
                        self.samples[k] = self.measurements[k][self.block_pos:self.block_pos + self.opt['block_size'], :]

                self.num_selected_samples = self.samples['positions'].shape[0]
                self.num_used_samples = self.num_selected_samples/(self.opt['skip_samples']+1)
            else:
                # simply use all data
                self.samples = self.measurements
            # rest of data selection is done from identification.py ATM, selected data will then be
            # in samples dict

        if self.opt['showTiming']:
            print("Loading samples from file took %.03f sec." % t.interval)

        self.inited = True

    def hasMoreSamples(self):
        """ tell if there are more samples to be added to the data used for identification """

        if not self.opt['selectBlocksFromMeasurements']:
            return False

        if self.block_pos + self.opt['block_size'] >= self.num_loaded_samples:
            return False

        return True

    def updateNumSamples(self):
        self.num_selected_samples = self.samples['positions'].shape[0]
        self.num_used_samples = self.num_selected_samples/(self.opt['skip_samples']+1)

    def removeLastSampleBlock(self):
        if self.opt['verbose']:
            print "removing block starting at {}".format(self.block_pos)
        for k in self.measurements.keys():
            self.samples[k] = np.delete(self.samples[k], range(self.num_selected_samples - self.opt['block_size'],
                                                               self.num_selected_samples), axis=0)
        self.updateNumSamples()
        if self.opt['verbose']:
            print "we now have {} samples selected (using {})".format(self.num_selected_samples, self.num_used_samples)

    def getNextSampleBlock(self):
        """ fill samples with next measurements block """

        # advance to next block or end of data
        self.block_pos += self.opt['block_size']

        if self.block_pos + self.opt['block_size'] > self.num_loaded_samples:
            self.opt['block_size'] = self.num_loaded_samples - self.block_pos

        if self.opt['verbose']:
            print "getting next block: {}/{}".format(self.block_pos, self.num_loaded_samples)

        for k in self.measurements.keys():
            if self.measurements[k].ndim == 0:
                mv = self.measurements[k]
            elif self.measurements[k].ndim == 1:
                mv = self.measurements[k][self.block_pos:self.block_pos + self.opt['block_size']]
            else:
                mv = self.measurements[k][self.block_pos:self.block_pos + self.opt['block_size'],:]
            self.samples[k] = mv

        self.updateNumSamples()

    def getBlockStats(self, model):
        """ check if we want to keep a new data block with the already selected ones """
        self.model = model

        # possible criteria for minimization:
        # * condition number of (base) regressor (+variations)
        # * largest per link condition number gets smaller (some are really huge though and are not
        # getting smaller with most new data)
        # * estimation error gets smaller (same data or validation)
        # ratio of min/max rel std devs

        # use condition number of regressor
        #new_condition_number = la.cond(self.YBase.dot(np.diag(self.xBaseModel)))   #weighted with a priori
        new_condition_number = la.cond(model.YBase)

        # get condition number for each of the links
        linkConds = model.getSubregressorsConditionNumbers()

        """
        # use largest link condition number
        largest_idx = np.argmax(linkConds)  #2+np.argmax(linkConds[2:7])
        new_condition_number = linkConds[largest_idx]
        """

        # use validation error
        # new_condition_number = self.val_error

        # use std dev ratio
        ##self.estimateRegressorTorques()
        """
        # get standard deviation of measurement and modeling error \sigma_{rho}^2
        rho = np.square(la.norm(self.tauMeasured-self.tauEstimated))
        sigma_rho = rho/(self.num_used_samples-self.model.num_base_params)

        # get standard deviation \sigma_{x} (of the estimated parameter vector x)
        C_xx = sigma_rho*(la.inv(np.dot(self.YBase.T, self.YBase)))
        sigma_x = np.diag(C_xx)

        # get relative standard deviation
        p_sigma_x = np.sqrt(sigma_x)
        for i in range(0, p_sigma_x.size):
            if np.abs(self.xBase[i]) != 0:
                p_sigma_x[i] /= np.abs(self.xBase[i])

        new_condition_number = np.max(p_sigma_x)/np.min(p_sigma_x)
        """

        self.seenBlocks.append((self.block_pos, self.opt['block_size'], new_condition_number, linkConds))

    def selectBlocks(self):
        """of all blocks loaded, select only those that create minimal condition number (cf. Venture, 2010)"""

        # select blocks with best 30% condition numbers
        perc_cond = np.percentile([cond for (b,bs,cond,linkConds) in self.seenBlocks], 50)

        cond_matrix = np.zeros((len(self.seenBlocks), self.model.N_LINKS))
        c = 0
        for block in self.seenBlocks:
            (b,bs,cond,linkConds) = block
            if cond > perc_cond:
                if self.opt['verbose']:
                    print "not using block starting at {} (cond {})".format(b, cond)
                self.unusedBlocks.append(block)
            else:
                if self.opt['verbose']:
                    print "using block starting at {} (cond {})".format(b, cond)
                self.usedBlocks.append(block)

                # create variance matrix
                cond_matrix[c, :] = linkConds
                c+=1

        ## look at sub-regressor patterns and throw out some similar blocks
        if self.opt['verbose']:
            print("checking for similar sub-regressor patterns")

        #check for pairs that are less than e.g. 15% of each other away
        #if found, delete larger one of the original blocks from usedBlocks (move to unused)
        #TODO: check this with the same file twice as input, should not use any blocks from the second file
        variances = np.var(cond_matrix[0:c,:],axis=1)
        v_idx = np.array(range(0, c))
        sort_idx = np.argsort(variances)

        to_delete = list()
        dist = 0.15
        i = 1
        while i < c:
            #keep two values of three close ones (only remove middle)
            #TODO: generalize to more values with this pattern
            if i<c-1 and np.abs(variances[sort_idx][i-1]-variances[sort_idx][i+1]) < np.abs(variances[sort_idx][i+1])*dist:
                to_delete.append(v_idx[sort_idx][i])
                i+=1
            #remove first if two are too close
            elif np.abs(variances[sort_idx][i-1]-variances[sort_idx][i]) < np.abs(variances[sort_idx][i])*dist:
                to_delete.append(v_idx[sort_idx][i-1])
            i+=1


        for d in np.sort(to_delete)[::-1]:
            if self.opt['verbose']:
                print "delete block {}".format(self.usedBlocks[d][0])
            del self.usedBlocks[d]


    def assembleSelectedBlocks(self):
        self.model.getSubregressorsConditionNumbers()
        if self.opt['verbose']:
            print("assembling selected blocks...\n")
        for k in self.measurements.keys():
            if not len(self.usedBlocks):
                break

            #init with first block
            (b, bs, cond, linkConds) = self.usedBlocks[0]
            if self.measurements[k].ndim == 0:
                self.samples[k] = self.measurements[k]
            else:
                self.samples[k] = self.measurements[k][b:b+bs]

            #append
            for i in range(1, len(self.usedBlocks)):
                (b, bs, cond, linkConds) = self.usedBlocks[i]
                if self.measurements[k].ndim == 0:
                    self.samples[k] = self.measurements[k]
                elif self.measurements[k].ndim == 1:
                    mv = self.measurements[k][b:b + bs]
                    #fix time offsets
                    mv = mv - mv[0] + (mv[1]-mv[0]) #let values start with first time diff
                    mv = mv + self.samples[k][-1]   #add after previous times
                    self.samples[k] = np.concatenate((self.samples[k], mv), axis=0)
                else:
                    mv = self.measurements[k][b:b + bs,:]
                    self.samples[k] = np.concatenate((self.samples[k], mv), axis=0)
        self.updateNumSamples()

    def removeZeroSamples(self):
        '''remove samples that have near zero velocity'''

        if self.opt['verbose']:
            print ("removing near zero samples..."),
        min_vel = 0.05   # rad per sec
        to_delete = list()
        for t in range(self.num_used_samples):
            if np.min(np.abs(self.samples['velocities'][t])) < min_vel:
                to_delete.append(t)

        for k in self.samples.keys():
            self.samples[k] = np.delete(self.samples[k], to_delete, 0)
        self.updateNumSamples()
        if self.opt['verbose']:
            print ("remaining samples: {}".format(self.num_used_samples))
