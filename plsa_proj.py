import numpy as np
import math


def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path, N, name_set):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.N = N
        self.name_set = name_set
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)

        self.background_prob_model = None #P(w/B) =>1D, just word axis
        self.lambda_b = 0.95

        self.topic_prob = None  # P(z=j | d, w) => 3D
        self.bg_prob = None  # P(z=B | d, w) => 2D, no topics axis

        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        # #############################
        for name_t in self.name_set:
            for file_no in range(1,self.N+1):
                with open(self.documents_path+name_t+"_"+str(file_no)+".txt") as f:
                    doc_temp = []
                    for line in f:
                        doc_temp = doc_temp + line.strip('\n').strip('\t').strip('.').strip(',').strip(' ').split(' ')
                    self.documents.append(doc_temp)
        self.number_of_documents = len(self.documents)
        # #############################

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        # #############################
        for d in self.documents:
            for w in d:
                if not(w in self.vocabulary):
                    self.vocabulary.append(w.lower())
        self.vocabulary_size = len(self.vocabulary)
        # #############################
        

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        # ############################
        self.term_doc_matrix = np.zeros((self.number_of_documents, self.vocabulary_size))
        
        for i in range(self.number_of_documents): # For each document in the corpus
            d = self.documents[i]
            word_count_dic = { word:0 for word in self.vocabulary}
            for w in d:                         # Count the occurance of each word in the vocab
                word_count_dic[w.lower()] += 1
            for w in word_count_dic.keys():     # Once counted, use the index of word in self.vocabulary as col_no(j) for self.term_doc_matrix[i][j]
                self.term_doc_matrix[i][self.vocabulary.index(w)] = word_count_dic[w]

        for i in range(self.number_of_documents):
            for j in range(self.vocabulary_size):
                if self.term_doc_matrix[i][j]==0:
                    self.term_doc_matrix[i][j]=0.001 #Smoothing

        # ############################

    def build_background_prob(self):
        self.build_term_doc_matrix()
        net_count = self.term_doc_matrix.sum(axis=0) #Add counts from all docs for every word
        self.background_prob = net_count/net_count.sum()
        ###############################

    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        # ############################
        self.document_topic_prob = normalize(np.random.rand(self.number_of_documents,number_of_topics)) # P(z | d) i.e pi_{d,j} = D -> One row per doc, col represent the topics
        # normalize such that sum of each row is 1 (sum of p(z/d) over all z for any given doc is 1)

        self.topic_word_prob = normalize(np.random.rand(number_of_topics,self.vocabulary_size)) # P(w | z) T -> Each row is a topic distribution, col are words in vocab
        # Normalize s.t each row (topic) is a distribution 
        # ############################

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        print("E step:")

        
        # ############################
        temp_T = np.repeat(self.topic_word_prob[np.newaxis,:, :], self.number_of_documents, axis=0)
        temp_D = np.repeat(self.document_topic_prob[:, :, np.newaxis], self.vocabulary_size, axis=2)
        # ############################
        temp_Z = np.multiply(temp_T, temp_D)

        temp_sum = temp_Z.sum(axis=1)

        ######Update p(z=j/d,w)##########
        self.topic_prob = temp_Z/temp_sum[:,np.newaxis,:]
        ################################

        #######Update P(z=B/d,w)#########
        temp_B = np.repeat(self.background_prob[np.newaxis,:], self.number_of_documents, axis=0)
        numerator = self.lambda_b*temp_B
        denominator = numerator + (1-self.lambda_b)*temp_sum

        self.bg_prob = numerator/denominator
        ##################################
            

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        print("M step:")

        temp_C = np.repeat(self.term_doc_matrix[:, np.newaxis, :], number_of_topics, axis=1)
        temp = np.multiply(temp_C,self.topic_prob)

        # update P(z | d)
        # ############################
        temp_D = temp.sum(axis=2) # add along the vocabulary axis
        self.document_topic_prob = normalize(temp_D)  # normalize along topic axis
        # ############################
        
        # update P(w | z)    
        # ############################
        temp_bg = np.repeat(self.bg_prob[:, np.newaxis, :], number_of_topics, axis=1)
        temp1 = np.multiply(temp, 1-temp_bg)
        temp_T = temp1.sum(axis=0)
        self.topic_word_prob = normalize(temp_T)
        #print(np.round(100*self.topic_word_prob))
        # ############################

        



    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        # ############################

        temp_T = np.repeat(self.topic_word_prob[np.newaxis,:, :], self.number_of_documents, axis=0)
        temp_D = np.repeat(self.document_topic_prob[:, :, np.newaxis], self.vocabulary_size, axis=2)
        temp_Z = np.multiply(temp_T, temp_D)
        temp_B = np.repeat(self.background_prob[np.newaxis,:], self.number_of_documents, axis=0)

        temp_netP = np.log( self.lambda_b*temp_B + (1-self.lambda_b)*temp_Z.sum(axis=1)) # net prob of a given word in given doc p(w/d) = sum over all z, p(w/d,z)*p(z)
        temp = np.multiply(self.term_doc_matrix,temp_netP)

        doc_likelihood = temp.sum(axis=1) # sum along word axis to get log liklihood of each doc

        # ############################
        
        return doc_likelihood.sum()

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = -np.inf

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")

            # ############################
            self.expectation_step()
            self.maximization_step(number_of_topics)
            likelihood = self.calculate_likelihood(number_of_topics)
            diff=likelihood-current_likelihood
            print(diff)
            if(diff < epsilon):
                break
            else:
                current_likelihood = likelihood
            # ############################



def main():
    documents_path = 'cnn/'
    N = 18 #no of docs
    name_set=['elon','bezos']
    corpus = Corpus(documents_path, N, name_set)
    corpus.build_corpus()
    corpus.build_vocabulary()
    corpus.build_background_prob()

    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 2
    max_iterations = 50
    epsilon = 0.001
    corpus.plsa(number_of_topics, max_iterations, epsilon)

    for topic_no in range(number_of_topics):
        a = corpus.topic_word_prob[topic_no,:]
        idx = sorted(range(len(a)), key=lambda i: a[i], reverse=True)[:30]
        print([corpus.vocabulary[j] for j in idx])

    a = corpus.background_prob
    idx = sorted(range(len(a)), key=lambda i: a[i], reverse=True)[:30]
    print([corpus.vocabulary[j] for j in idx])
    print([corpus.background_prob[j] for j in idx])



if __name__ == '__main__':
    main()
