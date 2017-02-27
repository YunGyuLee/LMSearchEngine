#include "porter2_stemmer.h"
#include <stdio.h>
#include <algorithm>
#include<string.h>

#include<vector>
#include<math.h>
#include<sstream>
#include<time.h>
#include<string> 
using namespace std;
#include <sstream>
#include <set>
#include <map>
#include <regex> //����ǥ���� ����� ���� include
#include<unordered_set>
#include<unordered_map>
#include <iomanip> //setw ����� ���� include //������ ���� ���ĸ��鶧 ��� �����߰��Ҷ�

#include <iostream>



using std::cout;
using std::endl;

#include <fstream>

using std::ifstream;
using std::ofstream;
#include <cstring>
const int Max_Chars_Per_Line = 512;
const int Max_Tokens_Per_Line = 20;//�� line�� �ִ� �ܾ��� ���� count�� token�� �ִ� ����
const char*const DELIMITER = " ";


struct cmp_str {
	bool operator()(char*first, char*second) {
		return strcmp(first, second)<0;
	}
};

//---------------------------------------------------������ ���� ����� �������� �ʿ��� �ڷᱸ��---------------------------------------------------


typedef std::unordered_set<string>WordList;  //��ü ���ξ���� ������ �ڷᱸ�� �ߺ��� ���ϱ� ���� set�� ���
WordList wordList;



typedef std::map<string, int>MAP2;
MAP2 DocFreq; //���ξ�ID�� DF�� mapping�� ������ �ڷᱸ��
MAP2 CollectionFreq; // CF�� ���ξ�ID�� mapping�� ������ �ڷᱸ��

typedef std::unordered_set<string>HASH; //���� Ž���� ���� unordered ���
HASH stopword; //�ҿ� ������ �ڷᱸ��
HASH wordSet; //�ϳ��� ������ ����� ���ξ���� ������ �ڷᱸ��

typedef std::unordered_multiset<string>MHASH; //�ش� ������ �ڷᱸ���� �״�� �Űܾ��ϴµ� ���� �ߺ��� �ʿ��� set�̹Ƿ� multiset���
MHASH oneDocument; //�ϳ��� ������ �о� �� �ȿ� ����� �ܾ���� ������ �ڷᱸ��


typedef std::map<string, vector<int>>MMAP; //key���� ���ξ��̰� �ش� value�� vector�� �̿�(���ʷ� docID�� TF�� ����)
MMAP indexInfo; //���ξ�� �ش� ����ID�� TF�� ã������ ���ξ key�� ������ �ش� key�� ���� ������ TF������ vector�� ���� 
				//���ʿ��� �޸� ����� ���̱� ���� -->multimap�� �̿������� �޸𸮸� �ʹ� ���� �����Ͽ� (�ϳ��� �ܾ ���� 
				//�ʹ��� ���� map�� ����� ���ʿ��� �޸𸮸� �Ҹ���)

typedef std::map<int, double>MAP;
MAP weightSum; //weight���� ����ID�� sum�� ������ �ڷᱸ��


//------------------------------------------------------�˻��� �ʿ��� �ڷᱸ��------------------------------------------------------------------------


typedef std::pair<string, double>Pair;	//value���� �ڷ����� �ٸ����� pair�� ������ vector�� �迭��
typedef std::unordered_map<int, Pair>DocFile; //doc file�� ����� ������ ������ �ڷᱸ�� ���� -->���� Ž���� ���� unordered ���
DocFile docFile; //key���� docID�� value�� string�� int�� pair�� ���� 

typedef std::unordered_map<string, vector<double>>WordFile; //-->�˻��� �ӵ��� ���� unordered ���
WordFile wordFile; //�ܾ� ���� ���Ͽ��� ������ �о� ������ �ڷᱸ�� //���ξ key�� ������ �� ���ξ�ID,DF,CF,���ξ� ������ġ �� value�� ������.

typedef std::map<string, double>MAP3;
MAP3 queryInfo;//query�ϳ��� ������ ����ִ� �ڷᱸ�� �ش� query�� ���ξ���� TF weight�� �����ϰ��ִ�. query�� ���� �Է¹����� ���� �ʱ�ȭ�ȴ�.

//typedef std::set<int>relevantDocList;
//relevantDocList relDocList; //relevant�� doclist�� ������ �ڷᱸ�� ���� ���� Ž���� �� �ʿ䰡 ���� ��ü�� �ѹ� �� �Ⱦ�� �ϱ� ������ set�� ���


typedef std::map<int, vector<double>>relevantDocList;
relevantDocList relDocList; //������ �������� �߷����� ���� �ڷᱸ�� key���� ����ID �̰� value�� ù��°�� cosine similarity�� �и� �ι�°�� ����, ����°�� query�ܾ ��Ÿ�� ����
							//��� �𵨿� ����� ���� key���� ���� ID �̰� value�� ù��°�� score , �ι�°�� �������� ������ ��� ��Ÿ������

typedef std::unordered_map<int ,double >Rank;
Rank relDocLGM;
Rank existOrNot; //query�� �ִ� �ܾ relDoc�� �����ϴ����� �����Ͽ� ��������� ������ �ڷᱸ��

WordList titleQ; //title�� �ִ� �ܾ���� ������ �ڷᱸ��

//--------------------------------recall - precision �׷����� ���� �ڷᱸ��-----------------------------
WordList Answer; //���� relevant Doc���� ������ �ڷᱸ��

typedef std::multimap<double, int>TEMP;
TEMP sortedResult;

int parsingAPW();
int parsingNYT();
int makestopword();

int indexing_file();
int calculateSum(); //weight���� �и� ���� ����ID ���� sum�� ������ ������ ����� �Լ�
void invert_indexing();
bool haveNum(string&);
void parse(string &);
int lengthOfNum(int);
int parsingQuery();
void searchByVSM();
int readyForSearch();
void searchByLanguageModel();
int testIndex();
int recall_precision();


regex pattern("([a-z]+)");

//double docWeight(int, int);
//void invert_indexing();

int totalDoc = 0;
double totalCF = 0;
template<typename A, typename B>
std::pair<B, A> flip_pair(const std::pair<A, B> &p)
{
	return std::pair<B, A>(p.second, p.first);
}

template<typename A, typename B>
std::multimap<B, A> flip_map(const std::map<A, B> &src)
{
	std::multimap<B, A> dst;
	std::transform(src.begin(), src.end(), std::inserter(dst, dst.begin()),
		flip_pair<A, B>);
	return dst;
}




template <typename T>
string NumToString(T pNumber);
int main() {
	clock_t before;
	double result;
	before = clock(); //�ð� ����� ���� ������

	
	
	makestopword();
	parsingQuery();
	

	


	
	
	readyForSearch();

	
	searchByLanguageModel();
	
	result = (double)(clock() - before) / CLOCKS_PER_SEC;
	printf("�� �ɸ��ð��� %5.2f �Դϴ�.\n", result);
	return 0;

}


int recall_precision() {
	ifstream fin1, fin2;
	fin1.open("relevant_document.txt");
	fin2.open("result.txt");

	if (!fin1.is_open()) {
		cout << "�������� ���½���" << endl;
		return -1;
	}
	if (!fin2.is_open()) {
		cout << "�������� ���½���" << endl;
		return -1;
	}

	double result[10];

	for (int i = 0; i < 10; i++) {
		result[i] = 0;
	}
	int num=303;
	while (!fin1.eof()) {
		
	     	string line;
			getline(fin1, line);
			char* cline = new char[line.length() + 1];
			strcpy(cline, line.c_str());

			if (line.empty()) {
				break;
			}

			const char*token[Max_Tokens_Per_Line] = {};

			token[0] = strtok(cline, "	");

			
			int n = 0;

			if (token[0]) {
				for (n = 1; n < Max_Tokens_Per_Line; n++) {



					token[n] = strtok(0, "	"); //""���� delimitor��� ��ūȭ

					if (!token[n])break;
				}
			}
			
					
		if (atoi(token[0]) != num) {
			
			int isFirst = 1;
			int countAns = 0;
			int count = 0;
			int ansPerTen = Answer.size() / 10;
			double interPol[10];

			for (int i = 0; i < 10; i++) {
				interPol[i] = 0;
			}
			
			while (!fin2.eof()) {
				string temp;
				getline(fin2, temp);
				
				if (isFirst == 1) {
					cout << "Query" << temp << " " << endl;
					isFirst = 0;
					continue;
				}
				else {


					if (temp.empty()) {
						cout << endl;
						break;

					}

					++count;
					if (Answer.find(temp) != Answer.end()) {
						++countAns;
						if (ansPerTen == countAns) {
							cout << "recall: " <<((double)ansPerTen / (double)Answer.size()) << " ";
							cout << "precision: " << (double)countAns / (double)count << endl;
							interPol[0] = (double)countAns / (double)count;
							
						}
						else if (2 * ansPerTen == countAns) {
							cout << "recall: " << ((double)2*ansPerTen / (double)Answer.size()) << " ";
							cout << "precision: " << (double)countAns / (double)count << endl;
							interPol[1] = (double)countAns / (double)count;
						}
						else if (3 * ansPerTen == countAns) {
							cout << "recall: " << ((double)3*ansPerTen / (double)Answer.size()) << " ";
							cout << "precision: " << (double)countAns / (double)count << endl;
							interPol[2] = (double)countAns / (double)count;
						}
						else if (4 * ansPerTen == countAns) {
							cout << "recall: " << ((double)4*ansPerTen / (double)Answer.size()) << " ";
							cout << "precision: " << (double)countAns / (double)count << endl;
							interPol[3] = (double)countAns / (double)count;
						}
						else if (5 * ansPerTen == countAns) {
							cout << "recall: " << ((double)5*ansPerTen / (double)Answer.size()) << " ";
							cout << "precision: " << (double)countAns / (double)count << endl;
							interPol[4] = (double)countAns / (double)count;
						}
						else if (6 * ansPerTen == countAns) {
							cout << "recall: " << ((double)6*ansPerTen / (double)Answer.size()) << " ";
							cout << "precision: " << (double)countAns / (double)count << endl;
							interPol[5] = (double)countAns / (double)count;
						}
						else if (7 * ansPerTen == countAns) {
							cout << "recall: " << ((double)7*ansPerTen / (double)Answer.size()) << " ";
							cout << "precision: " << (double)countAns / (double)count << endl;
							interPol[6] = (double)countAns / (double)count;
						}
						else if (8 * ansPerTen == countAns) {
							cout << "recall: " << ((double)8*ansPerTen / (double)Answer.size()) << " ";
							cout << "precision: " << (double)countAns / (double)count << endl;
							interPol[7] = (double)countAns / (double)count;
						}
						else if (9 * ansPerTen == countAns) {
							cout << "recall: " << ((double)9*ansPerTen / (double)Answer.size()) << " ";
							cout << "precision: " << (double)countAns / (double)count << endl;
							interPol[8] = (double)countAns / (double)count;
						}
						else if (Answer.size() == countAns) {
							cout << "recall: " << ((double)10*ansPerTen / (double)Answer.size()) << " ";
							cout << "precision: " << (double)countAns / (double)count << endl;
							interPol[9] = (double)countAns / (double)count;
						}


						


					}
					
					
				
				
				
				}
				



			}

			for (int i = 0; i < 10; i++) {
				if (interPol[i] == 0) {
					interPol[i] = interPol[i - 1];
				}
			}
			
			double maxP = 0;
			for (int i = 0; i < 10; i++) {
				maxP = interPol[i];
				for (int j = i+1; j < 10; j++) {
					if (interPol[i] <= interPol[j]) {
						maxP = interPol[j];
					}
				
				}
				result[i] = result[i] + maxP;
			}


			Answer.clear();
		}

		num = atoi(token[0]);
		string docID;
		docID = token[1];
		Answer.insert(docID);
		delete[] cline;
	}


	int isFirst = 1;
	int countAns = 0;
	int count = 0;
	int ansPerTen = Answer.size() / 10;
	double interPol[10];

	for (int i = 0; i < 10; i++) {
		interPol[i] = 0;
	}

	while (!fin2.eof()) {
		string temp;
		getline(fin2, temp);

		if (isFirst == 1) {
			cout << "Query" << temp << " " << endl;
			isFirst = 0;
			continue;
		}
		else {


			if (temp.empty()) {
				cout << endl;
				break;

			}

			++count;
			if (Answer.find(temp) != Answer.end()) {
				++countAns;
				if (ansPerTen == countAns) {
					cout << "recall: " << ((double)ansPerTen / (double)Answer.size()) << " ";
					cout << "precision: " << (double)countAns / (double)count << endl;
					interPol[0] = (double)countAns / (double)count;

				}
				else if (2 * ansPerTen == countAns) {
					cout << "recall: " << ((double)2 * ansPerTen / (double)Answer.size()) << " ";
					cout << "precision: " << (double)countAns / (double)count << endl;
					interPol[1] = (double)countAns / (double)count;
				}
				else if (3 * ansPerTen == countAns) {
					cout << "recall: " << ((double)3 * ansPerTen / (double)Answer.size()) << " ";
					cout << "precision: " << (double)countAns / (double)count << endl;
					interPol[2] = (double)countAns / (double)count;
				}
				else if (4 * ansPerTen == countAns) {
					cout << "recall: " << ((double)4 * ansPerTen / (double)Answer.size()) << " ";
					cout << "precision: " << (double)countAns / (double)count << endl;
					interPol[3] = (double)countAns / (double)count;
				}
				else if (5 * ansPerTen == countAns) {
					cout << "recall: " << ((double)5 * ansPerTen / (double)Answer.size()) << " ";
					cout << "precision: " << (double)countAns / (double)count << endl;
					interPol[4] = (double)countAns / (double)count;
				}
				else if (6 * ansPerTen == countAns) {
					cout << "recall: " << ((double)6 * ansPerTen / (double)Answer.size()) << " ";
					cout << "precision: " << (double)countAns / (double)count << endl;
					interPol[5] = (double)countAns / (double)count;
				}
				else if (7 * ansPerTen == countAns) {
					cout << "recall: " << ((double)7 * ansPerTen / (double)Answer.size()) << " ";
					cout << "precision: " << (double)countAns / (double)count << endl;
					interPol[6] = (double)countAns / (double)count;
				}
				else if (8 * ansPerTen == countAns) {
					cout << "recall: " << ((double)8 * ansPerTen / (double)Answer.size()) << " ";
					cout << "precision: " << (double)countAns / (double)count << endl;
					interPol[7] = (double)countAns / (double)count;
				}
				else if (9 * ansPerTen == countAns) {
					cout << "recall: " << ((double)9 * ansPerTen / (double)Answer.size()) << " ";
					cout << "precision: " << (double)countAns / (double)count << endl;
					interPol[8] = (double)countAns / (double)count;
				}
				else if (Answer.size() == countAns) {
					cout << "recall: " << ((double)10 * ansPerTen / (double)Answer.size()) << " ";
					cout << "precision: " << (double)countAns / (double)count << endl;
					interPol[9] = (double)countAns / (double)count;
				}





			}





		}




	}

	for (int i = 0; i < 10; i++) {
		if (interPol[i] == 0) {
			interPol[i] = interPol[i - 1];
		}
	}

	double maxP = 0;
	for (int i = 0; i < 10; i++) {
		maxP = interPol[i];
		for (int j = i + 1; j < 10; j++) {
			if (interPol[i] <= interPol[j]) {
				maxP = interPol[j];
			}

		}
		result[i] = result[i] + maxP;
	}


	Answer.clear();

	for (int i = 0; i < 10; i++) {
		cout << "recall: "<<0.1*(i+1) <<"precision: " << result[i] / 25 << endl;
	
	}


	if (fin1.is_open()) {
		fin1.close();
	}

	if (fin2.is_open()) {
		fin2.close();
	}

	return 1;
}

int testIndex() {
	ifstream fin;
	fin.open("Inverted_index.dat");
	if (!fin.is_open()) {
		cout << "������ ������ �� �� �����ϴ�." << endl;
		return -1;
	}
	else {
		int wordID;
		int docID;
		int TF;
		string weight;

		while (!fin.eof()) {
			fin >> wordID >> docID >> TF >> weight;
			
			if (weight.size() > 9) {
				cout<<weight.size()<<endl;
			}
			
		
		}

	
	}

	if (fin.is_open()) {
		fin.close();
	}
	return 0;
}

int readyForSearch() { //�ܾ� ���� ���ϰ� �������� ������ �޸𸮷� �ø��� ����
	
					   
	FILE* word; FILE* docu;
	fopen_s(&word, "term.dat", "r");
	fopen_s(&docu, "doc.dat", "r");
	char line[1000];
	char *token, *context = NULL;
	int i, a, s;

	string x;
	
	while (fgets(line, 1000, docu)) {
		
		token = strtok_r(line, "	", &context); i = atoi(token);
		
		x = strtok_r(context, "	", &context);
		
		token = strtok_r(context, "	", &context);
		
		a = atoi(token);
		
		
		docFile.insert(pair<int, pair<string, double>>(i, { x,(double)a }));
		
	}
	fclose(docu);
	while (fgets(line, 1000, word)) {
		token = strtok_s(line, "	", &context);
		x = strtok_s(context, "	", &context);
		token = strtok_s(context, "	", &context); i = atoi(token);
		token = strtok_s(context, "	", &context); a = atoi(token);
		token = strtok_s(context, "	", &context); s = atoi(token);
		wordFile.insert(pair<string, vector<double>>(x, { (double)i,(double)a,(double)s }));
		totalCF = totalCF + (double)a;
	}
	fclose(word);
	//				   ifstream fin;
	//fin.open("doc.dat");

	//if (!fin.is_open()) {
	//	cout << "�������� ���� ���� ����" << endl;
	//}
	//else {
	//	int docID;
	//	string docName;
	//	double docLength;
	//	while (!fin.eof()) {
	//		fin >> docID >> docName >> docLength;
	//		docFile.insert(std::make_pair(docID, std::make_pair(docName, docLength))); //pair value insert �ϴ� ���


	//	}
	//
	//}
	//
	//if (fin.is_open()) {
	//	fin.close();
	//}

	//fin.open("term.dat");
	//if (!fin.is_open()) {
	//	cout << "�ܾ����� ���� ���� ����" << endl;
	//}

	//else {
	//	double wordID;
	//	string word;
	//	double DF;
	//	double CF;
	//	double start;
	//	while (!fin.eof()) {
	//		fin >> wordID >> word >> DF>> CF >> start;
	//	   // wordFile[word].push_back(wordID);
	//		wordFile[word].push_back(DF);
	//		wordFile[word].push_back(CF);
	//		wordFile[word].push_back(start);
	//		totalCF = totalCF + CF;

	//	}

	//}

	//if (fin.is_open()) {
	//	fin.close();
	//}


}


void searchByLanguageModel() {
	//-----------------------------------------
	ifstream fin;
	ifstream query;
	ofstream fout;
    
	cout << "������ ����� �˻�����" << endl;
	fout.open("result_LGM.txt");
	fin.open("Inverted_index.dat");
	if (!fin.is_open()) {
		cout << "������ ���� ���� ����" << endl;
		return;
	}
	else {
		query.open("Query.txt");
		if (!query.is_open()) {
			cout << "���� ���� ���� ����" << endl;
			return;
		}
		else {
			size_t count = 0;
			int isTitle = 0;
			int isDesc = 0;
			string temp;
			//	int countFirst = 0;
			double u = 500000;
			while (!query.eof()) {
				//	query >> temp;
					//-------------------------�ٴ����� query �Է�--------------------------------------
				string line;

				getline(query, line);


				char* cline = new char[line.length() + 1];
				strcpy(cline, line.c_str());

				if (line.empty()) {//-------------query ���� �ڷᱸ�� ����----------------------
					//for (auto it = wordSet.begin(); it != wordSet.end(); ++it) {
					//	if (titleQ.find(*it) != titleQ.end()) { //�ش� �ܾ title�� ������ �ܾ���
					//		queryInfo.insert(std::make_pair(*it, 100 * oneDocument.count(*it)));
					//	
					//	}
					//	else {
					//		queryInfo.insert(std::make_pair(*it, oneDocument.count(*it)));
					//	}
					//}

					//wordSet.clear();
					//oneDocument.clear();
					//titleQ.clear();

					//-----------------------------------------------------------------------------
					
					
					int wordID;
					int docID;
					double TF;
					double weight;
					char buf[1000];
					string buf1;
					string temp1;
					
					auto compare = queryInfo.end();
					for (auto it = queryInfo.begin(); it != compare; ++it) {
						
							
							fin.clear();
							fin.seekg((size_t)(26 * wordFile[it->first][2])); //������ ��ġ�� ����Ű�� ��ġ�����ڰ� �׳� int�� �������� ������ �Ѿ ����������
							double DF = wordFile[it->first][0];
							for (int i = 0; i < DF; ++i) {
							//	fin >> wordID >> docID >> TF >> weight;
                                
                                
								fin.read(buf, 26);
								buf1 = buf;
								temp1 = buf1.substr(7, 14);
								docID =stoi(temp1);
								temp = buf1.substr(14, 17);
								TF = (double)(stoi(temp));

								
								existOrNot.insert(std::make_pair(docID, TF));


							
							}
						auto compare2 = relDocLGM.end();
						for (auto rel = relDocLGM.begin(); rel != compare2; rel++) {
							double tf = 0;

							if (existOrNot.find(rel->first) != existOrNot.end()) { //���� �ش� �ܾ relevant Doc�� �����ϸ�
								//auto tempTF = existOrNot.find(rel->first);
								tf = existOrNot[rel->first];
								//tf = tempTF->second;

							}

							rel->second = rel->second + log((it->second* tf + u * wordFile[it->first][1] / totalCF) / (docFile[docID].second + u));
															//tf�� ����ġ�� ������
							//if (relDocList.find(docID) == relDocList.end()) {
							//	double temp = log((TF + u * wordFile[it->first][2] / totalCF) / (docFile[docID].second + u));
							//	relDocList[docID].push_back(temp);//score�� ������ ����
							//	relDocList[docID].push_back(1); //������ ��Ÿ�� �ܾ count�ϱ� ���� �ʱ�ȭ

							//}
							//else {
							//	double temp = log((TF + u * wordFile[it->first][2] / totalCF) / (docFile[docID].second + u));
							//	relDocList[docID][0] = relDocList[docID][0] + temp;
							//	++relDocList[docID][1];//������ ��Ÿ�� �ܾ� �� count                        u��																								u��
							//}






						}
						existOrNot.clear();

					}

					auto compare3 = relDocLGM.end();
					for (auto it = relDocLGM.begin(); it != compare3; ++it) { //LGM�� ���� ������ rank�� ���� ���� �ϴ°��� --> multimap�� �̿��� �����ߴ�
																					  
						sortedResult.insert(std::make_pair(it->second, it->first));

					}


					auto it = sortedResult.end();
					--it;
					
					while (count < 1000) {


						++count;
						fout << docFile[it->second].first << endl;
						--it;
						
					}

					fout << endl;
					count = 0;
					queryInfo.clear();
					relDocLGM.clear();
					//rankingResult.clear(); //�ѹ� ������ �������� ���� ���� �ʱ�ȭ
					//count++;
					sortedResult.clear();

					
					cout << "�Ϸ�" << endl;
					continue;
				}


				const char*token[Max_Tokens_Per_Line] = {};

				token[0] = strtok(cline, DELIMITER);

				string temp;
				int n = 0;

				if (token[0]) {
					for (n = 1; n < Max_Tokens_Per_Line; n++) {



						token[n] = strtok(0, " ,-,/"); //""���� delimitor��� ��ūȭ

						if (!token[n])break;
					}
				}

				for (int i = 0; i < n; i++) {
					temp = token[i];



					if (temp.compare("QueryNum") == 0) {

						fout << token[1]<<endl;
						cout << "Query " << token[1] << " ����"<<endl;
						break;


					}
					else if ((temp.compare("[title]") == 0) || (isTitle == 1)) {

						if (temp.compare("[title]") == 0) {
							
							isTitle = 1;
							break;
						}
						int wordID;
						int docID;
						double TF;
						double weight;
						char buf[1000];
						string buf1;
						string temp1;

						fin.clear();
						fin.seekg((size_t)(26 * wordFile[temp][2])); //������ ��ġ�� ����Ű�� ��ġ�����ڰ� �׳� int�� �������� ������ �Ѿ ����������
						titleQ.insert(temp); //title�� ���Դ� �ܾ ������ �ڷᱸ��

						double DF = wordFile[temp][0];
					
						for (int i = 0; i < DF; ++i) {
							//fin >> wordID >> docID >> TF >> weight;
							
							fin.read(buf, 26);
							
							buf1 = buf;
							temp1 = buf1.substr(7, 14);
							docID = stoi(temp1);
							
							relDocLGM.insert(std::make_pair(docID, 0)); //relevant �� document ����
							

						}
						if (i == n - 1) {
							isTitle = 0;
						}
						
					}
					else if (temp.compare("[desc]") == 0||isDesc ==1) {
						if (temp.compare("[desc]") == 0) {
							if (relDocLGM.size() < 1000) {
								isDesc = 1;
							}
							break;
						}
						int wordID;
						int docID;
						double TF;
						double weight;
						char buf[1000];
						string buf1;
						string temp1;
						fin.clear();
						fin.seekg((size_t)(26 * wordFile[temp][2])); //������ ��ġ�� ����Ű�� ��ġ�����ڰ� �׳� int�� �������� ������ �Ѿ ����������
						

						double DF = wordFile[temp][0];
						for (int i = 0; i < DF; ++i) {
							//fin >> wordID >> docID >> TF >> weight;
							fin.read(buf, 26);
							buf1 = buf;
							temp1 = buf1.substr(7, 14);
							docID = stoi(temp1);
							relDocLGM.insert(std::make_pair(docID, 0)); //relevant �� document ����


						}
						if (i == n - 1) {
							isDesc = 0;
						}
						
					}
					
					
					if (titleQ.find(temp) != titleQ.end()) { //title�� �ִ� �ܾ�鿡 ����ġ
						queryInfo[temp] = queryInfo[temp] + 100;   
						continue;
					}
					queryInfo[temp] = queryInfo[temp] + 1; //�ƴϸ� ����ġ 1
					/*wordSet.insert(HASH::value_type(temp));
						oneDocument.insert(MHASH::value_type(temp));*/
					
				}
				delete cline;
			}



		}

		if (fin.is_open()) {
			fin.close();
		}
		if (fout.is_open()) {
			fout.close();
		}
		//�˻� ����





		//-----------------------------------------
	}
}

void searchByVSM() { //vector space model�� ����� �˻�
	ifstream fin;
	ifstream query;
	ofstream fout;
	cout << "Vector Space Model�� ����� �˻� ����" << endl;
	fout.open("result_VSM.txt");
	fin.open("Inverted_index.dat");
	if (!fin.is_open()) {
		cout << "������ ���� ���� ����" << endl;
		return;
	}
	else {
		query.open("Query.txt");
		if (!query.is_open()) {
			cout << "���� ���� ���� ����" << endl;
			return;
		}
		else {
//			int count = 0;
//			string temp;
//			int countFirst=0;
//			while (!query.eof()) {
//				query >> temp;
//				if ((temp.compare("QueryNum") == 0)||(countFirst==1)) {
//					
//					
//					
//					
//					if (countFirst == 1) {
//						fout <<temp<< endl;
//						countFirst = 0;
//						fout << endl;
//						cout << "Query " + temp + " �Ϸ�" << endl;
//						continue;
//					}
//					//-------------------------------------query ���� �ڷᱸ�� ����--------------------------------
//					
//					if (wordSet.size() != 0) {
//						
//
//						/*if (count != 1) {
//							cout << wordSet.size() << endl;
//							
//						}
//*/
//						for (auto it = wordSet.begin(); it != wordSet.end(); ++it) {
//
//							queryInfo.insert(std::make_pair(*it, oneDocument.count(*it)));
//
//						}
//
//						wordSet.clear();
//						oneDocument.clear();
//
//						//------------------------------------relevant�� �������� ����---------------------------------
//
//						int wordID;
//						int docID;
//						int TF;
//						double weight;
//
//						for (auto it = queryInfo.begin(); it != queryInfo.end(); ++it) { //��� �ϳ��� ������ ����
//							
//							fin.clear();
//							fin.seekg(static_cast<size_t>(26 * wordFile[it->first][2]));
//							for (int i = 0; i < wordFile[it->first][0]; ++i) {
//								fin >> wordID >> docID >> TF >> weight;
//
//								
//
//								//-----------------------�ش� ������ query�ܾ ��� �����ߴ°��� count �� ��ü query�ܾ� ��� �ش� ������ ������ 
//								//----------------------query�ܾ��� ������ �̿��� relevant�� ������ ��������.
//								
//								if (relDocList.find(docID) == relDocList.end()) {
//									relDocList[docID].push_back(weight*(it->second));//���ڸ� ������ ����
//									relDocList[docID].push_back(weight*weight); //�и� ������ ����
//									relDocList[docID].push_back(1);//������ ��Ÿ�� query�ܾ� ���� count
//								}
//								else {
//									relDocList[docID][0] = relDocList[docID][0] + weight*(it->second);
//									relDocList[docID][1] = relDocList[docID][1] + weight*weight;
//									++relDocList[docID][2];
//								}
//
//
//							}
//
//
//						}
//
//						/*
//						if (count != 1) {
//							cout << relDocList.size() << endl;
//
//						}*/
//
//						for (auto it = relDocList.begin(); it != relDocList.end(); ++it) {
//
//								if(static_cast<double>(it->second[2]/queryInfo.size()) > 0.2)
//								sortedResult.insert(std::make_pair(relDocList[it->first][0] / sqrt(relDocList[it->first][1]),it->first ));
//							
//						}
//
//						/*if (count != 1) {
//							cout << rankingResult.size() << endl;
//
//						}*/
//						
//						//sortedResult = flip_map(rankingResult);
//						
//						for (auto it = sortedResult.end(); count < 1000; --it) {
//
//							if (it == sortedResult.end()) {
//								continue;
//							}
//
//							++count;
//							fout << docFile[it->second].first << endl;
//
//						}
//							
//						count = 0; //���� 1000�� ����� ���� ���� �ʱ�ȭ
//						queryInfo.clear();
//						relDocList.clear();
//						sortedResult.clear(); //�ѹ� ������ �������� ���� ���� �ʱ�ȭ
//						//count++;
//					}
//					//---------------------------------------------------------------------------
//					
//					//fout << temp << " ";
//					countFirst = 1;
//					
//					continue;
//				}
//				
//
//
//				wordSet.insert(HASH::value_type(temp));
//				oneDocument.insert(MHASH::value_type(temp));
//			
//			
//			}
//		
//		//--------------------------------������ ���� ���� ���-------------------------------
//
//
//		/*if (count != 1) {
//		cout << wordSet.size() << endl;
//
//		}
//		*/
//			for (auto it = wordSet.begin(); it != wordSet.end(); ++it) {
//
//				queryInfo.insert(std::make_pair(*it, oneDocument.count(*it)));
//
//			}
//
//			wordSet.clear();
//			oneDocument.clear();
//
//			//------------------------------------relevant�� �������� ����---------------------------------
//
//			int wordID;
//			int docID;
//			int TF;
//			double weight;
//
//			for (auto it = queryInfo.begin(); it != queryInfo.end(); ++it) { //��� �ϳ��� ������ ����
//
//				fin.clear();
//				fin.seekg(static_cast<size_t>(26 * wordFile[it->first][2]));
//				for (int i = 0; i < wordFile[it->first][0]; ++i) {
//					fin >> wordID >> docID >> TF >> weight;
//
//					
//
//					//-----------------------�ش� ������ query�ܾ ��� �����ߴ°��� count �� ��ü query�ܾ� ��� �ش� ������ ������ 
//					//----------------------query�ܾ��� ������ �̿��� relevant�� ������ ��������.
//
//					if (relDocList.find(docID) == relDocList.end()) {
//						relDocList[docID].push_back(weight*(it->second));//���ڸ� ������ ����
//						relDocList[docID].push_back(weight*weight); //�и� ������ ����
//						relDocList[docID].push_back(1);//������ ��Ÿ�� query�ܾ� ���� count
//					}
//					else {
//						relDocList[docID][0] = relDocList[docID][0] + weight*(it->second);
//						relDocList[docID][1] = relDocList[docID][1] + weight*weight;
//						++relDocList[docID][2];
//					}
//
//
//				}
//
//
//			}
//
//			/*
//			if (count != 1) {
//			cout << relDocList.size() << endl;
//
//			}*/
//
//			for (auto it = relDocList.begin(); it != relDocList.end(); ++it) {
//				if (static_cast<double>(it->second[2] / queryInfo.size()) > 0.2)
//					sortedResult.insert(std::make_pair(relDocList[it->first][0] / sqrt(relDocList[it->first][1]) ,it->first ));
//
//			}
//
//			/*if (count != 1) {
//			cout << rankingResult.size() << endl;
//
//			}*/
//
//		//	sortedResult = flip_map(rankingResult);
//			for (auto it = sortedResult.end(); count < 1000; --it) {
//
//				if (it == sortedResult.end()) {
//					continue;
//				}
//
//				++count;
//				fout << docFile[it->second].first << endl;
//
//			}
//
//			count = 0;
//			queryInfo.clear();
//			relDocList.clear();
//			sortedResult.clear();
//		
//		
int count = 0;
int isTitle = 0;
int isDesc = 0;
string temp;
//	int countFirst = 0;

while (!query.eof()) {
	//	query >> temp;
	//-------------------------�ٴ����� query �Է�--------------------------------------
	string line;

	getline(query, line);


	char* cline = new char[line.length() + 1];
	strcpy(cline, line.c_str());

	if (line.empty()) {//-------------query ���� �ڷᱸ�� ����----------------------
					   //for (auto it = wordSet.begin(); it != wordSet.end(); ++it) {
					   //	if (titleQ.find(*it) != titleQ.end()) { //�ش� �ܾ title�� ������ �ܾ���
					   //		queryInfo.insert(std::make_pair(*it, 100 * oneDocument.count(*it)));
					   //	
					   //	}
					   //	else {
					   //		queryInfo.insert(std::make_pair(*it, oneDocument.count(*it)));
					   //	}
					   //}

					   //wordSet.clear();
					   //oneDocument.clear();
					   //titleQ.clear();

					   //-----------------------------------------------------------------------------
		

		int wordID;
		int docID;
		double TF;
		double weight;

		for (auto it = queryInfo.begin(); it != queryInfo.end(); ++it) {

			fin.clear();
			fin.seekg(static_cast<size_t>(26 * wordFile[it->first][2])); //������ ��ġ�� ����Ű�� ��ġ�����ڰ� �׳� int�� �������� ������ �Ѿ ����������
			double DF = wordFile[it->first][0];
			for (int i = 0; i < DF; ++i) {
				fin >> wordID >> docID >> TF >> weight;

				if (relDocList.find(docID) != relDocList.end()) {
					relDocList[docID][0] = relDocList[docID][0] + weight*(it->second);
					relDocList[docID][1] = relDocList[docID][1] + weight*weight;
				}

				
			}
			

		}


		for (auto it = relDocList.begin(); it != relDocList.end(); ++it) { //VSM�� ���� ������ rank�� ���� ���� �ϴ°��� --> multimap�� �̿��� �����ߴ�

			sortedResult.insert(std::make_pair(relDocList[it->first][0] / sqrt(relDocList[it->first][1]), it->first));

		}


		auto it = sortedResult.end();
		--it;
		while (count < 1000) {


			++count;
			fout << docFile[it->second].first << endl;
			--it;
		}


		count = 0;
		queryInfo.clear();
		relDocList.clear();
		//rankingResult.clear(); //�ѹ� ������ �������� ���� ���� �ʱ�ȭ
		//count++;
		sortedResult.clear();


		cout << "�Ϸ�" << endl;
		continue;
	}


	const char*token[Max_Tokens_Per_Line] = {};
    
    
    
	token[0] = strtok(cline, DELIMITER);

	string temp;
	int n = 0;

	if (token[0]) {
		for (n = 1; n < Max_Tokens_Per_Line; n++) {



			token[n] = strtok(0, " ,-,/"); //""���� delimitor��� ��ūȭ

			if (!token[n])break;
		}
	}

	for (int i = 0; i < n; i++) {
		temp = token[i];



		if (temp.compare("QueryNum") == 0) {

			fout << token[1] << endl;
			cout << "Query " << token[1] << " ����" << endl;
			break;


		}
		else if ((temp.compare("[title]") == 0) || (isTitle == 1)) {

			if (temp.compare("[title]") == 0) {
				isTitle = 1;
				break;
			}
			int wordID;
			int docID;
			double TF;
			double weight;
			fin.clear();
			fin.seekg(static_cast<size_t>(26 * wordFile[temp][2])); //������ ��ġ�� ����Ű�� ��ġ�����ڰ� �׳� int�� �������� ������ �Ѿ ����������
			titleQ.insert(temp); //title�� ���Դ� �ܾ ������ �ڷᱸ��
			double DF = wordFile[temp][0];
			for (int i = 0; i < DF; ++i) {
				fin >> wordID >> docID >> TF >> weight;
				if (relDocList.find(docID) == relDocList.end()) {
					relDocList[docID].push_back(0);//���ڸ� ������ ����
					relDocList[docID].push_back(0); //�и� ������ ����
					
				}
				


			}
			if (i == n - 1) {
				isTitle = 0;
			}
		}
		else if (temp.compare("[desc]") == 0) {
			break;


		}
		else if (temp.compare("[narr]") == 0) {
			break;

		}

		
		queryInfo[temp] = queryInfo[temp] + 1; //�ƴϸ� 
												 

	}
	delete cline;
}




		//-------------------------------------------------------------------------------------
		
		}

		if (query.is_open()) {
			query.close();
		}


	}

	if (fin.is_open()) {
		fin.close();
	}
	if (fout.is_open()) {
		fout.close();
	}
	//�˻� ����

}


int parsingQuery() {  //�Է����� ���� ���������� parsing�Ͽ� query���·� �ٲٴ� �Լ�
	
	ifstream fin;
	ofstream fout;
	fin.open("topics25.txt");
	fout.open("Query.txt");
	if (!fin.is_open()) {
		cout << "Query ������ ���� ���߽��ϴ�." << endl;
		return -1;
	}
	else {
		
		string temp;
		int isNarr = 0;
		while (!fin.eof()) {
		
			
			string line;

			getline(fin, line);

			if (line.empty())continue;
			char* cline = new char[line.length() + 1];
			strcpy(cline, line.c_str());


			int n = 0;
			bool isEmpty = false; //�ش� ���� ������� �Ⱥ������ Ȯ���ϴ� ����
			const char*token[Max_Tokens_Per_Line] = {};

			token[0] = strtok(cline, DELIMITER);

			string temp;


			if (token[0]) {
				for (n = 1; n<Max_Tokens_Per_Line; n++) {



					token[n] = strtok(0, " ,-,/"); //""���� delimitor��� ��ūȭ

					if (!token[n])break;
				}
			}


			for (int i = 0; i<n; i++) {
				temp = token[i];
				parse(temp);


				if (strcmp(token[0], "<num>") == 0) {
					fout << "QueryNum ";
					fout << token[i + 2] << endl;
					break;

				}
				else if (strcmp(token[0], "<desc>") == 0) {
					fout << "[desc] " << endl;
					break;
					
				}
				else if (strcmp(token[0], "<narr>") == 0) {//modified.
					//fout << "[narr] " << endl;
					if (strcmp(token[0], "<narr>") == 0) {
						isNarr = 1;
					}//modified
					break;

				}
				else if (strcmp(token[i], "<title>") == 0) {
					fout << "[title] " << endl;
					continue;
				}
				else {
					
					

					if (strcmp(token[0], "</top>") == 0) {
						fout << "\n";
						isNarr = 0;//modified.
						
						break;
					}
					if (isNarr == 1) {
						continue;
					}

					HASH::iterator iter = stopword.find(temp);
					if (iter != stopword.end()) { //�ҿ������
						if ((i == n - 1) && (i != 0) && isEmpty) {
							fout << "\n";

						}
						continue;
					}



					if (haveNum(temp)) { //���� �� ���ʿ��� ���� ���Խ� ����

						if ((i == n - 1) && (i != 0) && isEmpty) {
							fout << "\n";

						}

						continue;//---------------
					}
					Porter2Stemmer::stem(temp); //stemming �۾�
					iter = stopword.find(temp); 
					if (iter != stopword.end()) { //stemming �� �ٽ��ѹ� �ҿ�� ����
						if ((i == n - 1) && (i != 0) && isEmpty) { //���� �ش����� ������ �ܾ��̰� �ش� �ٿ� �ܾ �ϳ��� �����ϸ� ���� �� continue
							fout << "\n";

						}
						continue;
					}
					fout << temp << " ";
					isEmpty = true;
					if (i == n - 1) {

						fout << "\n";
					}
				}


			}

			delete cline;

		
		
		
		
		}
	
	
	
	}

	if (fin.is_open()) {
		fin.close();
	
	}
	if (fout.is_open()) {
		fout.close();
	
	}
	cout << "Query ���� �۾� �Ϸ�." << endl;
	return 0;
}

bool haveNum(string & word) {  //string�� ���ĺ��̿��� ���ڸ� ������ ������ true �ƴϸ� false


	if (!regex_match(word, pattern)) {
		return true;
	}
	else
		return false;

}
void parse(std::string &word) { //�ҹ��ڷ� �ٲٰ� ���ʿ��� ���ڵ��� ���� �ϴ� �Լ�

	std::transform(word.begin(), word.end(), word.begin(), ::tolower);



	word.erase(std::remove(word.begin(), word.end(), '`'), word.end());
	word.erase(std::remove(word.begin(), word.end(), '?'), word.end());
	word.erase(std::remove(word.begin(), word.end(), '('), word.end());
	word.erase(std::remove(word.begin(), word.end(), ')'), word.end());
	word.erase(std::remove(word.begin(), word.end(), ','), word.end());
	word.erase(std::remove(word.begin(), word.end(), '_'), word.end());
	word.erase(std::remove(word.begin(), word.end(), '-'), word.end());
	word.erase(std::remove(word.begin(), word.end(), '!'), word.end());


	Porter2Stemmer::internal::replaceIfExists(word, "''", "", 0);
	Porter2Stemmer::internal::replaceIfExists(word, "'''", "", 0);
	Porter2Stemmer::internal ::replaceIfExists(word, ".", "", 0);
	Porter2Stemmer::internal ::replaceIfExists(word, ";", "", 0);
	Porter2Stemmer::internal ::replaceIfExists(word, ":", "", 0);



}

/*void search(string str){
HASH::iterator iter = stopword.find(str);
if(iter !=stopword.end()){
cout<<"key: "<<str<<endl;
}else{
cout<<"��ġ�ϴ� ���� �����ϴ�."<<endl;
}

}*/
template<typename T>
string NumToString(T pNumber) {
	ostringstream oOStrStream;
	oOStrStream << pNumber;
	return oOStrStream.str();

}
int lengthOfNum(int num) {
	string length;

	length = NumToString(num);
	return length.size();


}

int calculateSum() {
	
	int docIDtoCompare = 1; //���Ͽ� �ִ� ���� ID�� �񱳸� ���� ����
	double sum = 0; //weight����� ���� �и� ����ϱ� ���� ����
	ifstream Indexed;
	ofstream fout;

	fout.open("sum.dat");
	Indexed.open("indexing.dat");
	if (!Indexed.is_open()) {
		cout << "���������� ���� ���߽��ϴ�." << endl;
		return 1;
	}
	else {
		cout << "�������� ���� ����" << endl;


		int docID;
		string word;
		int TF;

		while (!Indexed.eof()) {
			Indexed >> docID >> word >> TF;
			if (!(docID == docIDtoCompare)) {


				sum = sqrt(sum);

				weightSum.insert(std::make_pair(docID, sum));
					fout << docIDtoCompare << " " << sum << "\n";
				docIDtoCompare = docID;
				sum = 0;


			}

			auto it = DocFreq.find(word);


			sum = sum + pow((log((double)(TF)) + 1)*log(((double)totalDoc / (double)it->second)), 2.0);

		}

		


	}

	if (Indexed.is_open()) {
		Indexed.close();
	}
	if (fout.is_open()) {
	fout.close();
	}

	cout << "weight ����� ���� �и�(sum)�� ���� �Ϸ�" << endl;
	
}


void invert_indexing() {
	
	int wordID = 1; //���ξ� ID�ο��� ���� ����
	
	double totalRate = 0;
	double weight; //���� docWeight�� ������ ����
	ifstream fin;

	ofstream foutI;
	ofstream foutT;
	foutI.open("Inverted_index.dat");
	foutT.open("term.dat");
	fin.open("indexing.dat");

	int starting = 0; //���������� ������ġ�� ������ ����
	
	cout << "������ ���� �ۼ���..." << endl;
	
	for (auto its = wordList.begin(); its != wordList.end(); ++its) {
		
		auto it = DocFreq.find(*its);

			for (size_t i = 0; i < indexInfo[*its].size(); i = i + 2) {

			auto sum = weightSum.find(indexInfo[*its][i]);
			weight = ((log((double)(indexInfo[*its][i+1]))+1) * log(((double)totalDoc / (double)it->second))) / sum->second;
			if (isinf(weight) == 1) {
				weight = 0;
			}




			/* int blank1 = 7 - lengthOfNum(wordID);
			int blank2 = 7 - lengthOfNum(info->second.first);
			int blank3 = 3 - lengthOfNum(info->second.second);
			int blank4 = 3 - lengthOfNum(weight);*/

			foutI << setw(7) << wordID; //setw(n) �̶� wordID�� 7byte�� �����ϰ� ���� �κ��� �������� ä���
			foutI << setw(7) << indexInfo[*its][i];  //���� ID 
			foutI << setw(3) << indexInfo[*its][i+1]; //TF
			foutI << fixed << showpoint << setw(9) << setprecision(4) << weight; //setprecision : �Ҽ��� �ڸ������ߴ� �Լ� 
			//���ÿ��� fix -> showpoint -> setw -> setprecision ������ ���!!


		}


		auto temp = CollectionFreq.find(*its);

		foutT << wordID << "	" << *its << "	" << it->second << "	" << temp->second << "	" << starting << endl;
		starting = starting + it->second;
		wordID++;


		//
	}


	if (foutI.is_open()) {
		foutI.close();
	}
	if (foutT.is_open()) {
		foutT.close();
	}
	if (fin.is_open()) {
		fin.close();
	}

	cout << "����� 2������ �������Ͽ���. ����ߴ�." << endl;

}

int indexing_file() {
	ifstream fin;
	ofstream foutD;
	ofstream foutI;
	ofstream foutT;

	int year = 0;
	int month = 0;
	int Max_day = 0;
	int Max_month = 0;




	string docName;
	foutD.open("doc.dat");
	foutI.open("indexing.dat");
	foutT.open("term.dat");

	for (year = 1998; year <= 2000; year++) {
		if (year == 1998) {
			month = 6;
			Max_month = 12;
		}
		if (year == 1999) {
			month = 1;
			Max_month = 12;
		}
		if (year == 2000) {
			month = 1;
			Max_month = 9;
		}


		string strM;
		string strD;


		for (month; month <= Max_month; month++) {
			if (year == 1998) {
				if (month == 6 || month == 9 || month == 11) {
					Max_day = 30;
				}
				else if (month == 7 || month == 8 || month == 10 || month == 12) {
					Max_day = 31;
				}
			}
			if (year == 1999) {
				if (month == 4 || month == 6 || month == 9 || month == 11) {
					Max_day = 30;
				}
				else if (month == 1 || month == 3 || month == 5 || month == 7 || month == 8 || month == 10 || month == 12) {
					Max_day = 31;
				}
				else {
					Max_day = 28;
				}


			}

			if (year == 2000) {
				if (month == 4 || month == 6 || month == 9) {
					Max_day = 30;
				}
				else if (month == 1 || month == 3 || month == 5 || month == 7 || month == 8) {
					Max_day = 31;
				}
				else {
					Max_day = 29;
				}

			}

			if (month<10) {
				strM = "0" + NumToString(month);
			}
			else {
				strM = NumToString(month);
			}
			for (int day = 1; day <= Max_day; day++) {
				cout << day << endl;
				if (day<10) {
					strD = "0" + NumToString(day);
				}
				else {
					strD = NumToString(day);
				}

				fin.open("ParsedNYT/" + NumToString(year) + "/p" + NumToString(year) + strM + strD + "_NYT.txt");
				cout << "ParsedNYT/" + NumToString(year) + "/p" + NumToString(year) + strM + strD + "_NYT.txt Indexing����" << endl;


				if (!fin.is_open()) {
					cout << "������ ���� ���߽��ϴ�." << endl;
					//continue;
				}

			/*		if (!(year == 1998) || month > 6) {
				return 1;
				}*/
				else {
					while (!fin.eof()) {

						//char buf[Max_Chars_Per_Line];

						//fin.getline(buf,Max_Chars_Per_Line);

						string line;



						getline(fin, line);

						char* cline = new char[line.length() + 1];
						strcpy(cline, line.c_str());


						int n = 0;

						const char*token[Max_Tokens_Per_Line] = {};

						token[0] = strtok(cline, DELIMITER);




						string temp;



						if (line.empty()) {
							foutD << totalDoc << "	";
							foutD << docName << "	";
							foutD << oneDocument.size() << "\n";


							for (auto it = wordSet.begin(); it != wordSet.end(); ++it) {
								foutI << totalDoc << "	";
								foutI << *it << "	";
								foutI << oneDocument.count(*it) << "\n";
								indexInfo[*it].push_back(totalDoc); //docID �ֱ�
								indexInfo[*it].push_back(oneDocument.count(*it)); //TF �ֱ�


								CollectionFreq[*it] = CollectionFreq[*it] + oneDocument.count(*it); //CF�� TF���� ��
								DocFreq[*it]++; //DF ����

												//			TermFreq.insert(pair<string, int>(NumToString(docID)+ "_"+ NumToString(it2->second), oneDocument.count(*it)));

												//			word_doc.insert(pair<int, int>(it2->second, docID));

							}

							wordSet.clear();
							oneDocument.clear();


						}


						if (token[0]) {
							for (n = 1; n<Max_Tokens_Per_Line; n++) {
								token[n] = strtok(0, DELIMITER);
								if (!token[n])break;
							}
						}
						for (int i = 0; i<n; i++) {
							temp = token[i];

							if (strcmp(token[0], "[DOCNO]") == 0) {


								docName = token[i + 2];

								totalDoc++;
								break;

							}
							else if (strcmp(token[0], "[HEADLINE]") == 0 || strcmp(token[i], ":") == 0 || strcmp(token[0], "[TEXT]") == 0) {

								continue;

							}


							else {
								wordSet.insert(HASH::value_type(temp));
								oneDocument.insert(MHASH::value_type(temp));
								//  voca.insert(HASH::value_type(temp));
								/*word_wordID.insert(pair<string, int>(temp,voca.size()));
								DocFreq.insert(pair<int, int>(voca.size(), 0));
								CollectionFreq.insert(pair<int, int>(voca.size(), 0));*/
								wordList.insert(temp);

								DocFreq.insert(pair<string, int>(temp, 0));
								CollectionFreq.insert(pair<string, int>(temp, 0));
							}


						}

						delete cline;
					}
					cout << "NYT/" + NumToString(year) + "/" + NumToString(year) + strM + strD + "_NYT �Ϸ�" << endl;


					if (fin.is_open()) {
						fin.close();
					}

				}
				fin.close();
			}
		}
	}
	//
	for (year = 1998; year <= 2000; year++) {
		if (year == 1998) {
			month = 6;
			Max_month = 12;
		}
		if (year == 1999) {
			month = 1;
			Max_month = 11;
		}
		if (year == 2000) {
			month = 1;
			Max_month = 9;
		}


		string strM;
		string strD;
		for (month; month <= Max_month; month++) {

			if (year == 1998) {
				if (month == 6 || month == 9 || month == 11) {
					Max_day = 30;
				}
				else if (month == 7 || month == 8 || month == 10 || month == 12) {
					Max_day = 31;
				}
			}
			if (year == 1999) {
				if (month == 4 || month == 6 || month == 9) {
					Max_day = 30;
				}
				else if (month == 1 || month == 3 || month == 5 || month == 7 || month == 8 || month == 10) {
					Max_day = 31;
				}
				else if (month == 2) {
					Max_day = 28;
				}
				else {
					Max_day = 1;
				}
			}


			if (year == 2000) {
				if (month == 4 || month == 6 || month == 9) {
					Max_day = 30;
				}
				else if (month == 1 || month == 3 || month == 5 || month == 7 || month == 8) {
					Max_day = 31;
				}
				else {
					Max_day = 29;
				}

			}

			if (month<10) {
				strM = "0" + NumToString(month);
			}
			else {
				strM = NumToString(month);
			}
			for (int day = 1; day <= Max_day; day++) {
				cout << day << endl;
				if (day<10) {
					strD = "0" + NumToString(day);
				}
				else {
					strD = NumToString(day);
				}

				fin.open("ParsedAPW/" + NumToString(year) + "/p" + NumToString(year) + strM + strD + "_APW_ENG.txt");
				cout << "ParsedAPW/" + NumToString(year) + "/p" + NumToString(year) + strM + strD + "_APW_ENG.txt Indexing����" << endl;


				if (!fin.is_open()) {
					cout << "������ ���� ���߽��ϴ�." << endl;
					//continue;
				}

				/*	if (!(year == 1998) || month > 6) {
				return 1;
				}*/
				else {
					while (!fin.eof()) {

						//char buf[Max_Chars_Per_Line];

						//fin.getline(buf,Max_Chars_Per_Line);

						string line;



						getline(fin, line);

						char* cline = new char[line.length() + 1];
						strcpy(cline, line.c_str());


						int n = 0;

						const char*token[Max_Tokens_Per_Line] = {};

						token[0] = strtok(cline, DELIMITER);




						string temp;



						if (line.empty()) {
							foutD << totalDoc << "	";
							foutD << docName << "	";
							foutD << oneDocument.size() << "\n";


							for (auto it = wordSet.begin(); it != wordSet.end(); ++it) {
								foutI << totalDoc << "	";
								foutI << *it << "	";
								foutI << oneDocument.count(*it) << "\n";
								indexInfo[*it].push_back(totalDoc);
								indexInfo[*it].push_back(oneDocument.count(*it));


								CollectionFreq[*it] = CollectionFreq[*it] + oneDocument.count(*it); //CF�� TF���� ��
								DocFreq[*it]++; //DF ���� 

												//			TermFreq.insert(pair<string, int>(NumToString(docID)+ "_"+ NumToString(it2->second), oneDocument.count(*it)));

												//			word_doc.insert(pair<int, int>(it2->second, docID));

							}

							wordSet.clear();
							oneDocument.clear();


						}


						if (token[0]) {
							for (n = 1; n<Max_Tokens_Per_Line; n++) {
								token[n] = strtok(0, DELIMITER);
								if (!token[n])break;
							}
						}
						for (int i = 0; i<n; i++) {
							temp = token[i];

							if (strcmp(token[0], "[DOCNO]") == 0) {


								docName = token[i + 2];

								totalDoc++;
								break;

							}
							else if (strcmp(token[0], "[HEADLINE]") == 0 || strcmp(token[i], ":") == 0 || strcmp(token[0], "[TEXT]") == 0) {

								continue;

							}


							else {
								wordSet.insert(HASH::value_type(temp));
								oneDocument.insert(MHASH::value_type(temp));
								//  voca.insert(HASH::value_type(temp));
								/*word_wordID.insert(pair<string, int>(temp,voca.size()));
								DocFreq.insert(pair<int, int>(voca.size(), 0));
								CollectionFreq.insert(pair<int, int>(voca.size(), 0));*/
								wordList.insert(temp);

								DocFreq.insert(pair<string, int>(temp, 0));
								CollectionFreq.insert(pair<string, int>(temp, 0));
							}


						}

						delete cline;
					}
					cout << "NYT/" + NumToString(year) + "/" + NumToString(year) + strM + strD + "_NYT �Ϸ�" << endl;


					if (fin.is_open()) {
						fin.close();
					}

				}
				fin.close();
			}
		}
	}//
	 //
	if (foutD.is_open()) {
		foutD.close();
	}
	if (foutI.is_open()) {
		foutI.close();
	}
	if (foutT.is_open()) {
		foutT.close();
	}
	cout << "indexing�Ϸ�" << endl;
	return 1;


}
int makestopword() {


	ifstream fin;

	fin.open("stopword.txt");

	if (!fin.is_open()) {
		cout << "������ ���� ���߽��ϴ�." << endl;
		return 1;
	}
	else {

		while (!fin.eof()) {

			string line;

			getline(fin, line);


			if (!line.empty()) {
				stopword.insert(HASH::value_type(line));

			}
			else {
				continue;
			}


		}
	}

	if (fin.is_open()) {
		fin.close();
	}

}

int parsingNYT() {
	ifstream fin;
	ofstream fout;


	int year = 0;
	int month = 0;
	int Max_day = 0;
	int Max_month = 0;
	for (year = 1998; year <= 2000; year++) {
		if (year == 1998) {
			month = 6;
			Max_month = 12;
		}
		if (year == 1999) {
			month = 1;
			Max_month = 12;
		}
		if (year == 2000) {
			month = 1;
			Max_month = 9;
		}


		string strM;
		string strD;
		for (month; month <= Max_month; month++) {
			if (year == 1998) {
				if (month == 6 || month == 9 || month == 11) {
					Max_day = 30;
				}
				else if (month == 7 || month == 8 || month == 10 || month == 12) {
					Max_day = 31;
				}
			}
			if (year == 1999) {
				if (month == 4 || month == 6 || month == 9 || month == 11) {
					Max_day = 30;
				}
				else if (month == 1 || month == 3 || month == 5 || month == 7 || month == 8 || month == 10 || month == 12) {
					Max_day = 31;
				}
				else {
					Max_day = 28;
				}


			}

			if (year == 2000) {
				if (month == 4 || month == 6 || month == 9) {
					Max_day = 30;
				}
				else if (month == 1 || month == 3 || month == 5 || month == 7 || month == 8) {
					Max_day = 31;
				}
				else {
					Max_day = 29;
				}

			}

			if (month<10) {
				strM = "0" + NumToString(month);
			}
			else {
				strM = NumToString(month);
			}
			for (int day = 1; day <= Max_day; day++) {
				cout << day << endl;
				if (day<10) {
					strD = "0" + NumToString(day);
				}
				else {
					strD = NumToString(day);
				}

				fin.open("NYT/" + NumToString(year) + "/" + NumToString(year) + strM + strD + "_NYT");
				cout << "NYT/" + NumToString(year) + "/" + NumToString(year) + strM + strD + "_NYT ����" << endl;
				fout.open("ParsedNYT/" + NumToString(year) + "/p" + NumToString(year) + strM + strD + "_NYT.txt");
				int checkH = 0;
				int checkT = 0;
				int checkEnd = 0;
				if (!fin.is_open()) {
					cout << "������ ���� ���߽��ϴ�." << endl;
					continue;
				}
				else {
					while (!fin.eof()) {

						//char buf[Max_Chars_Per_Line];

						//fin.getline(buf,Max_Chars_Per_Line);

						string line;

						getline(fin, line);

						if (line.empty())continue;
						char* cline = new char[line.length() + 1];
						strcpy(cline, line.c_str());


						int n = 0;
						bool isEmpty = false;
						const char*token[Max_Tokens_Per_Line] = {};

						token[0] = strtok(cline, DELIMITER);

						string temp;


						if (token[0]) {
							for (n = 1; n<Max_Tokens_Per_Line; n++) {



								token[n] = strtok(0, " ,-");

								if (!token[n])break;
							}
						}


						for (int i = 0; i<n; i++) {
							temp = token[i];
							parse(temp);


							if (strcmp(token[0], "<DOCNO>") == 0) {
								fout << "[DOCNO] : ";
								fout << token[i + 1] << endl;
								break;

							}
							else if (strcmp(token[0], "<HEADLINE>") == 0 || checkH == 1) {

								if (checkH == 0) {
									fout << "[HEADLINE] : ";
									checkH = 1;
									continue;
								}


								HASH::iterator iter = stopword.find(temp);
								if (iter != stopword.end()) {
									continue;
								}
								if (strcmp(token[i], "</HEADLINE>") == 0) {
									fout << endl;
									checkH = 0;
									break;

								}
								if (haveNum(temp))continue;

								Porter2Stemmer::stem(temp);
								iter = stopword.find(temp);
								if (iter != stopword.end()) {
									continue;
								}

								fout << temp << " ";


							}
							else if (strcmp(token[0], "<TEXT>") == 0 || checkT == 1) {
								if (checkT == 0) {
									fout << "[TEXT] : " << endl;
									checkT = 1;
									break;
								}
								if (strcmp(token[0], "</TEXT>") == 0) {
									fout << "\n";

									checkT = 0;
									break;
								}

								HASH::iterator iter = stopword.find(temp);
								if (iter != stopword.end()) {
									if ((i == n - 1) && (i != 0) && isEmpty) {
										fout << "\n";

									}
									continue;
								}



								if (haveNum(temp)) {

									if ((i == n - 1) && (i != 0) && isEmpty) {
										fout << "\n";

									}

									continue;//---------------
								}
								Porter2Stemmer::stem(temp);
								iter = stopword.find(temp);
								if (iter != stopword.end()) {
									if ((i == n - 1) && (i != 0) && isEmpty) {
										fout << "\n";

									}
									continue;
								}
								fout << temp << " ";
								isEmpty = true;
								if ((i == n - 1)) {

									fout << "\n";
								}
							}
							else {
								break;
							}


						}

						delete cline;
					}
					cout << "NYT/" + NumToString(year) + "/" + NumToString(year) + strM + strD + "_NYT �Ϸ�" << endl;

					if (fout.is_open()) {
						fout.close();
					}
					if (fin.is_open()) {
						fin.close();
					}

				}

			}
		}
	}
	cout << "NYT stemming �Ϸ�" << endl;
	return 0;
}
int parsingAPW() {
	ifstream fin;
	ofstream fout;


	int year = 0;
	int month = 0;
	int Max_day = 0;
	int Max_month = 0;
	for (year = 1998; year <= 2000; year++) {
		if (year == 1998) {
			month = 6;
			Max_month = 12;
		}
		if (year == 1999) {
			month = 1;
			Max_month = 11;
		}
		if (year == 2000) {
			month = 1;
			Max_month = 9;
		}


		string strM;
		string strD;
		for (month; month <= Max_month; month++) {

			if (year == 1998) {
				if (month == 6 || month == 9 || month == 11) {
					Max_day = 30;
				}
				else if (month == 7 || month == 8 || month == 10 || month == 12) {
					Max_day = 31;
				}
			}
			if (year == 1999) {
				if (month == 4 || month == 6 || month == 9) {
					Max_day = 30;
				}
				else if (month == 1 || month == 3 || month == 5 || month == 7 || month == 8 || month == 10) {
					Max_day = 31;
				}
				else if (month == 2) {
					Max_day = 28;
				}
				else {
					Max_day = 1;
				}
			}


			if (year == 2000) {
				if (month == 4 || month == 6 || month == 9) {
					Max_day = 30;
				}
				else if (month == 1 || month == 3 || month == 5 || month == 7 || month == 8) {
					Max_day = 31;
				}
				else {
					Max_day = 29;
				}

			}

			if (month<10) {
				strM = "0" + NumToString(month);
			}
			else {
				strM = NumToString(month);
			}
			for (int day = 1; day <= Max_day; day++) {
				cout << day << endl;
				if (day<10) {
					strD = "0" + NumToString(day);
				}
				else {
					strD = NumToString(day);
				}

				fin.open("APW/" + NumToString(year) + "/" + NumToString(year) + strM + strD + "_APW_ENG");
				cout << "APW/" + NumToString(year) + "/" + NumToString(year) + strM + strD + "_APW_ENG ����" << endl;
				fout.open("ParsedAPW/" + NumToString(year) + "/p" + NumToString(year) + strM + strD + "_APW_ENG.txt");
				int checkH = 0;
				int checkT = 0;
				int checkEnd = 0;
				if (!fin.is_open()) {
					cout << "������ ���� ���߽��ϴ�." << endl;
					continue;
				}
				else {
					while (!fin.eof()) {

						//char buf[Max_Chars_Per_Line];

						//fin.getline(buf,Max_Chars_Per_Line);

						string line;

						getline(fin, line);

						if (line.empty())continue;
						char* cline = new char[line.length() + 1];
						strcpy(cline, line.c_str());


						int n = 0;
						bool isEmpty = false;
						const char*token[Max_Tokens_Per_Line] = {};

						token[0] = strtok(cline, DELIMITER);

						string temp;


						if (token[0]) {
							for (n = 1; n<Max_Tokens_Per_Line; n++) {



								token[n] = strtok(0, " ,-");

								if (!token[n])break;
							}
						}


						for (int i = 0; i<n; i++) {
							temp = token[i];
							parse(temp);


							if (strcmp(token[0], "<DOCNO>") == 0) {
								fout << "[DOCNO] : ";
								fout << token[i + 1] << endl;
								break;

							}
							else if (strcmp(token[0], "<HEADLINE>") == 0 || checkH == 1) {

								if (checkH == 0) {
									fout << "[HEADLINE] : ";
									checkH = 1;
									continue;
								}


								HASH::iterator iter = stopword.find(temp);
								if (iter != stopword.end()) {
									continue;
								}
								if (strcmp(token[i], "</HEADLINE>") == 0) {
									fout << endl;
									checkH = 0;
									break;

								}
								if (haveNum(temp))continue;

								Porter2Stemmer::stem(temp);
								iter = stopword.find(temp);
								if (iter != stopword.end()) {
									continue;
								}

								fout << temp << " ";


							}
							else if (strcmp(token[0], "<TEXT>") == 0 || checkT == 1) {
								if (checkT == 0) {
									fout << "[TEXT] : " << endl;
									checkT = 1;
									break;
								}
								if (strcmp(token[0], "</TEXT>") == 0) {
									fout << "\n";

									checkT = 0;
									break;
								}

								HASH::iterator iter = stopword.find(temp);
								if (iter != stopword.end()) {
									if ((i == n - 1) && (i != 0) && isEmpty) {
										fout << "\n";

									}
									continue;
								}



								if (haveNum(temp)) {

									if ((i == n - 1) && (i != 0) && isEmpty) {
										fout << "\n";

									}

									continue;//---------------
								}
								Porter2Stemmer::stem(temp);
								iter = stopword.find(temp);
								if (iter != stopword.end()) {
									if ((i == n - 1) && (i != 0) && isEmpty) {
										fout << "\n";

									}
									continue;
								}
								fout << temp << " ";
								isEmpty = true;
								if (i == n - 1) {

									fout << "\n";
								}
							}
							else {
								break;
							}


						}

						delete[] cline;
					}
					//cout<<fin.eof()<<endl;

				}
				cout << "APW/" + NumToString(year) + "/" + NumToString(year) + strM + strD + "_APW_ENG �Ϸ�" << endl;

				if (fout.is_open()) {
					fout.close();
				}
				if (fin.is_open()) {
					fin.close();
				}

			}

		}
	}


	cout << "APW stemming �Ϸ�" << endl;
	return 0;
}
