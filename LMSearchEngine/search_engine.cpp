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
#include <regex> //정규표현식 사용을 위한 include
#include<unordered_set>
#include<unordered_map>
#include <iomanip> //setw 사용을 위한 include //역색인 파일 형식만들때 사용 공백추가할때

#include <iostream>



using std::cout;
using std::endl;

#include <fstream>

using std::ifstream;
using std::ofstream;
#include <cstring>
const int Max_Chars_Per_Line = 512;
const int Max_Tokens_Per_Line = 20;//각 line에 있는 단어의 수를 count할 token의 최대 개수
const char*const DELIMITER = " ";


struct cmp_str {
	bool operator()(char*first, char*second) {
		return strcmp(first, second)<0;
	}
};

//---------------------------------------------------역색인 파일 만드는 과정에서 필요한 자료구조---------------------------------------------------


typedef std::unordered_set<string>WordList;  //전체 색인어들을 저장할 자료구조 중복을 피하기 위해 set을 사요
WordList wordList;



typedef std::map<string, int>MAP2;
MAP2 DocFreq; //색인어ID와 DF를 mapping해 저장할 자료구조
MAP2 CollectionFreq; // CF를 색인어ID와 mapping해 저장할 자료구조

typedef std::unordered_set<string>HASH; //빠른 탐색을 위해 unordered 사용
HASH stopword; //불용어를 저장할 자료구조
HASH wordSet; //하나의 문서에 저장된 색인어들을 저장할 자료구조

typedef std::unordered_multiset<string>MHASH; //해당 문서를 자료구조로 그대로 옮겨야하는데 값의 중복이 필요한 set이므로 multiset사용
MHASH oneDocument; //하나의 문서를 읽어 그 안에 저장된 단어들을 저장할 자료구조


typedef std::map<string, vector<int>>MMAP; //key값이 색인어이고 해당 value로 vector를 이용(차례로 docID와 TF를 넣음)
MMAP indexInfo; //색인어로 해당 문서ID와 TF를 찾기위한 색인어를 key로 가지고 해당 key에 관한 문서와 TF정보를 vector에 저장 
				//불필요한 메모리 사용을 줄이기 위해 -->multimap을 이용했지만 메모리를 너무 많이 차지하여 (하나의 단어에 대해 
				//너무나 많은 map을 만들어 불필요한 메모리를 소모함)

typedef std::map<int, double>MAP;
MAP weightSum; //weight들의 문서ID별 sum을 저장할 자료구조


//------------------------------------------------------검색에 필요한 자료구조------------------------------------------------------------------------


typedef std::pair<string, double>Pair;	//value값의 자료형이 다를때는 pair로 같으면 vector나 배열로
typedef std::unordered_map<int, Pair>DocFile; //doc file에 저장된 정보를 저장할 자료구조 생성 -->빠른 탐색을 위해 unordered 사용
DocFile docFile; //key값을 docID로 value는 string과 int의 pair로 구성 

typedef std::unordered_map<string, vector<double>>WordFile; //-->검색의 속도를 위해 unordered 사용
WordFile wordFile; //단어 정보 파일에서 정보를 읽어 저장할 자료구조 //색인어를 key로 가지고 각 색인어ID,DF,CF,색인어 시작위치 를 value로 가진다.

typedef std::map<string, double>MAP3;
MAP3 queryInfo;//query하나의 정보를 담고있는 자료구조 해당 query의 색인어들의 TF weight를 저장하고있다. query를 새로 입력받을때 마다 초기화된다.

//typedef std::set<int>relevantDocList;
//relevantDocList relDocList; //relevant한 doclist를 저장할 자료구조 굳이 빠른 탐색을 할 필요가 없고 전체를 한번 다 훑어야 하기 때문에 set을 사용


typedef std::map<int, vector<double>>relevantDocList;
relevantDocList relDocList; //적합한 문서들을 추려내기 위한 자료구조 key값은 문서ID 이고 value의 첫번째는 cosine similarity의 분모 두번째는 분자, 세번째는 query단어가 나타난 개수
							//언어 모델에 사용할 때는 key값이 문서 ID 이고 value의 첫번째는 score , 두번째는 쿼리텀이 문서에 몇번 나타났는지

typedef std::unordered_map<int ,double >Rank;
Rank relDocLGM;
Rank existOrNot; //query에 있는 단어가 relDoc에 존재하는지를 저장하여 점수계산을 도와줄 자료구조

WordList titleQ; //title에 있는 단어들을 저장할 자료구조

//--------------------------------recall - precision 그래프를 위한 자료구조-----------------------------
WordList Answer; //정답 relevant Doc들을 저장할 자료구조

typedef std::multimap<double, int>TEMP;
TEMP sortedResult;

int parsingAPW();
int parsingNYT();
int makestopword();

int indexing_file();
int calculateSum(); //weight에서 분모를 구해 문서ID 별로 sum을 저장할 파일을 만드는 함수
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
	before = clock(); //시간 계산을 위한 변수들

	
	
	makestopword();
	parsingQuery();
	

	


	
	
	readyForSearch();

	
	searchByLanguageModel();
	
	result = (double)(clock() - before) / CLOCKS_PER_SEC;
	printf("총 걸린시간은 %5.2f 입니다.\n", result);
	return 0;

}


int recall_precision() {
	ifstream fin1, fin2;
	fin1.open("relevant_document.txt");
	fin2.open("result.txt");

	if (!fin1.is_open()) {
		cout << "정답파일 오픈실패" << endl;
		return -1;
	}
	if (!fin2.is_open()) {
		cout << "정답파일 오픈실패" << endl;
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



					token[n] = strtok(0, "	"); //""안의 delimitor들로 토큰화

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
		cout << "역색인 파일을 열 수 없습니다." << endl;
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

int readyForSearch() { //단어 정보 파일과 문서정보 파일을 메모리로 올리는 과정
	
					   
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
	//	cout << "문서정보 파일 오픈 실패" << endl;
	//}
	//else {
	//	int docID;
	//	string docName;
	//	double docLength;
	//	while (!fin.eof()) {
	//		fin >> docID >> docName >> docLength;
	//		docFile.insert(std::make_pair(docID, std::make_pair(docName, docLength))); //pair value insert 하는 방법


	//	}
	//
	//}
	//
	//if (fin.is_open()) {
	//	fin.close();
	//}

	//fin.open("term.dat");
	//if (!fin.is_open()) {
	//	cout << "단어정보 파일 오픈 실패" << endl;
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
    
	cout << "언어모델을 사용한 검색시작" << endl;
	fout.open("result_LGM.txt");
	fin.open("Inverted_index.dat");
	if (!fin.is_open()) {
		cout << "역색인 파일 오픈 실패" << endl;
		return;
	}
	else {
		query.open("Query.txt");
		if (!query.is_open()) {
			cout << "쿼리 파일 오픈 실패" << endl;
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
					//-------------------------줄단위로 query 입력--------------------------------------
				string line;

				getline(query, line);


				char* cline = new char[line.length() + 1];
				strcpy(cline, line.c_str());

				if (line.empty()) {//-------------query 정보 자료구조 생성----------------------
					//for (auto it = wordSet.begin(); it != wordSet.end(); ++it) {
					//	if (titleQ.find(*it) != titleQ.end()) { //해당 단어가 title에 등장한 단어라면
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
							fin.seekg((size_t)(26 * wordFile[it->first][2])); //파일의 위치를 가리키는 위치지정자가 그냥 int로 했을때는 범위가 넘어가 오류가생김
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

							if (existOrNot.find(rel->first) != existOrNot.end()) { //만약 해당 단어가 relevant Doc에 존재하면
								//auto tempTF = existOrNot.find(rel->first);
								tf = existOrNot[rel->first];
								//tf = tempTF->second;

							}

							rel->second = rel->second + log((it->second* tf + u * wordFile[it->first][1] / totalCF) / (docFile[docID].second + u));
															//tf에 가중치를 곱해줌
							//if (relDocList.find(docID) == relDocList.end()) {
							//	double temp = log((TF + u * wordFile[it->first][2] / totalCF) / (docFile[docID].second + u));
							//	relDocList[docID].push_back(temp);//score를 저장할 공간
							//	relDocList[docID].push_back(1); //문서에 나타난 단어를 count하기 위해 초기화

							//}
							//else {
							//	double temp = log((TF + u * wordFile[it->first][2] / totalCF) / (docFile[docID].second + u));
							//	relDocList[docID][0] = relDocList[docID][0] + temp;
							//	++relDocList[docID][1];//문서에 나타난 단어 수 count                        u값																								u값
							//}






						}
						existOrNot.clear();

					}

					auto compare3 = relDocLGM.end();
					for (auto it = relDocLGM.begin(); it != compare3; ++it) { //LGM로 만든 점수를 rank에 따라 정렬 하는과정 --> multimap을 이용해 정렬했다
																					  
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
					//rankingResult.clear(); //한번 수행이 끝났으니 관련 정보 초기화
					//count++;
					sortedResult.clear();

					
					cout << "완료" << endl;
					continue;
				}


				const char*token[Max_Tokens_Per_Line] = {};

				token[0] = strtok(cline, DELIMITER);

				string temp;
				int n = 0;

				if (token[0]) {
					for (n = 1; n < Max_Tokens_Per_Line; n++) {



						token[n] = strtok(0, " ,-,/"); //""안의 delimitor들로 토큰화

						if (!token[n])break;
					}
				}

				for (int i = 0; i < n; i++) {
					temp = token[i];



					if (temp.compare("QueryNum") == 0) {

						fout << token[1]<<endl;
						cout << "Query " << token[1] << " 시작"<<endl;
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
						fin.seekg((size_t)(26 * wordFile[temp][2])); //파일의 위치를 가리키는 위치지정자가 그냥 int로 했을때는 범위가 넘어가 오류가생김
						titleQ.insert(temp); //title에 나왔던 단어를 저장할 자료구조

						double DF = wordFile[temp][0];
					
						for (int i = 0; i < DF; ++i) {
							//fin >> wordID >> docID >> TF >> weight;
							
							fin.read(buf, 26);
							
							buf1 = buf;
							temp1 = buf1.substr(7, 14);
							docID = stoi(temp1);
							
							relDocLGM.insert(std::make_pair(docID, 0)); //relevant 한 document 삽입
							

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
						fin.seekg((size_t)(26 * wordFile[temp][2])); //파일의 위치를 가리키는 위치지정자가 그냥 int로 했을때는 범위가 넘어가 오류가생김
						

						double DF = wordFile[temp][0];
						for (int i = 0; i < DF; ++i) {
							//fin >> wordID >> docID >> TF >> weight;
							fin.read(buf, 26);
							buf1 = buf;
							temp1 = buf1.substr(7, 14);
							docID = stoi(temp1);
							relDocLGM.insert(std::make_pair(docID, 0)); //relevant 한 document 삽입


						}
						if (i == n - 1) {
							isDesc = 0;
						}
						
					}
					
					
					if (titleQ.find(temp) != titleQ.end()) { //title에 있는 단어들에 가중치
						queryInfo[temp] = queryInfo[temp] + 100;   
						continue;
					}
					queryInfo[temp] = queryInfo[temp] + 1; //아니면 가중치 1
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
		//검색 종료





		//-----------------------------------------
	}
}

void searchByVSM() { //vector space model을 사용한 검색
	ifstream fin;
	ifstream query;
	ofstream fout;
	cout << "Vector Space Model을 사용한 검색 시작" << endl;
	fout.open("result_VSM.txt");
	fin.open("Inverted_index.dat");
	if (!fin.is_open()) {
		cout << "역색인 파일 오픈 실패" << endl;
		return;
	}
	else {
		query.open("Query.txt");
		if (!query.is_open()) {
			cout << "쿼리 파일 오픈 실패" << endl;
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
//						cout << "Query " + temp + " 완료" << endl;
//						continue;
//					}
//					//-------------------------------------query 정보 자료구조 생성--------------------------------
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
//						//------------------------------------relevant한 문서집합 추출---------------------------------
//
//						int wordID;
//						int docID;
//						int TF;
//						double weight;
//
//						for (auto it = queryInfo.begin(); it != queryInfo.end(); ++it) { //적어도 하나의 쿼리텀 포함
//							
//							fin.clear();
//							fin.seekg(static_cast<size_t>(26 * wordFile[it->first][2]));
//							for (int i = 0; i < wordFile[it->first][0]; ++i) {
//								fin >> wordID >> docID >> TF >> weight;
//
//								
//
//								//-----------------------해당 문서가 query단어를 몇개나 포함했는가를 count 후 전체 query단어 대비 해당 문서가 포함한 
//								//----------------------query단어의 비율을 이용해 relevant한 문서를 가려낸다.
//								
//								if (relDocList.find(docID) == relDocList.end()) {
//									relDocList[docID].push_back(weight*(it->second));//분자를 저장할 공간
//									relDocList[docID].push_back(weight*weight); //분모를 저장할 공간
//									relDocList[docID].push_back(1);//문서에 나타난 query단어 개수 count
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
//						count = 0; //상위 1000개 출력을 위한 변수 초기화
//						queryInfo.clear();
//						relDocList.clear();
//						sortedResult.clear(); //한번 수행이 끝났으니 관련 정보 초기화
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
//		//--------------------------------마지막 쿼리 정보 출력-------------------------------
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
//			//------------------------------------relevant한 문서집합 추출---------------------------------
//
//			int wordID;
//			int docID;
//			int TF;
//			double weight;
//
//			for (auto it = queryInfo.begin(); it != queryInfo.end(); ++it) { //적어도 하나의 쿼리텀 포함
//
//				fin.clear();
//				fin.seekg(static_cast<size_t>(26 * wordFile[it->first][2]));
//				for (int i = 0; i < wordFile[it->first][0]; ++i) {
//					fin >> wordID >> docID >> TF >> weight;
//
//					
//
//					//-----------------------해당 문서가 query단어를 몇개나 포함했는가를 count 후 전체 query단어 대비 해당 문서가 포함한 
//					//----------------------query단어의 비율을 이용해 relevant한 문서를 가려낸다.
//
//					if (relDocList.find(docID) == relDocList.end()) {
//						relDocList[docID].push_back(weight*(it->second));//분자를 저장할 공간
//						relDocList[docID].push_back(weight*weight); //분모를 저장할 공간
//						relDocList[docID].push_back(1);//문서에 나타난 query단어 개수 count
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
	//-------------------------줄단위로 query 입력--------------------------------------
	string line;

	getline(query, line);


	char* cline = new char[line.length() + 1];
	strcpy(cline, line.c_str());

	if (line.empty()) {//-------------query 정보 자료구조 생성----------------------
					   //for (auto it = wordSet.begin(); it != wordSet.end(); ++it) {
					   //	if (titleQ.find(*it) != titleQ.end()) { //해당 단어가 title에 등장한 단어라면
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
			fin.seekg(static_cast<size_t>(26 * wordFile[it->first][2])); //파일의 위치를 가리키는 위치지정자가 그냥 int로 했을때는 범위가 넘어가 오류가생김
			double DF = wordFile[it->first][0];
			for (int i = 0; i < DF; ++i) {
				fin >> wordID >> docID >> TF >> weight;

				if (relDocList.find(docID) != relDocList.end()) {
					relDocList[docID][0] = relDocList[docID][0] + weight*(it->second);
					relDocList[docID][1] = relDocList[docID][1] + weight*weight;
				}

				
			}
			

		}


		for (auto it = relDocList.begin(); it != relDocList.end(); ++it) { //VSM로 만든 점수를 rank에 따라 정렬 하는과정 --> multimap을 이용해 정렬했다

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
		//rankingResult.clear(); //한번 수행이 끝났으니 관련 정보 초기화
		//count++;
		sortedResult.clear();


		cout << "완료" << endl;
		continue;
	}


	const char*token[Max_Tokens_Per_Line] = {};
    
    
    
	token[0] = strtok(cline, DELIMITER);

	string temp;
	int n = 0;

	if (token[0]) {
		for (n = 1; n < Max_Tokens_Per_Line; n++) {



			token[n] = strtok(0, " ,-,/"); //""안의 delimitor들로 토큰화

			if (!token[n])break;
		}
	}

	for (int i = 0; i < n; i++) {
		temp = token[i];



		if (temp.compare("QueryNum") == 0) {

			fout << token[1] << endl;
			cout << "Query " << token[1] << " 시작" << endl;
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
			fin.seekg(static_cast<size_t>(26 * wordFile[temp][2])); //파일의 위치를 가리키는 위치지정자가 그냥 int로 했을때는 범위가 넘어가 오류가생김
			titleQ.insert(temp); //title에 나왔던 단어를 저장할 자료구조
			double DF = wordFile[temp][0];
			for (int i = 0; i < DF; ++i) {
				fin >> wordID >> docID >> TF >> weight;
				if (relDocList.find(docID) == relDocList.end()) {
					relDocList[docID].push_back(0);//분자를 저장할 공간
					relDocList[docID].push_back(0); //분모를 저장할 공간
					
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

		
		queryInfo[temp] = queryInfo[temp] + 1; //아니면 
												 

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
	//검색 종료

}


int parsingQuery() {  //입력으로 받은 주제파일을 parsing하여 query형태로 바꾸는 함수
	
	ifstream fin;
	ofstream fout;
	fin.open("topics25.txt");
	fout.open("Query.txt");
	if (!fin.is_open()) {
		cout << "Query 파일을 열지 못했습니다." << endl;
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
			bool isEmpty = false; //해당 줄이 비었는지 안비었는지 확인하는 변수
			const char*token[Max_Tokens_Per_Line] = {};

			token[0] = strtok(cline, DELIMITER);

			string temp;


			if (token[0]) {
				for (n = 1; n<Max_Tokens_Per_Line; n++) {



					token[n] = strtok(0, " ,-,/"); //""안의 delimitor들로 토큰화

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
					if (iter != stopword.end()) { //불용어제거
						if ((i == n - 1) && (i != 0) && isEmpty) {
							fout << "\n";

						}
						continue;
					}



					if (haveNum(temp)) { //숫자 및 불필요한 문자 포함시 제거

						if ((i == n - 1) && (i != 0) && isEmpty) {
							fout << "\n";

						}

						continue;//---------------
					}
					Porter2Stemmer::stem(temp); //stemming 작업
					iter = stopword.find(temp); 
					if (iter != stopword.end()) { //stemming 후 다시한번 불용어 제거
						if ((i == n - 1) && (i != 0) && isEmpty) { //만약 해당줄의 마지막 단어이고 해당 줄에 단어가 하나라도 존재하면 개행 후 continue
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
	cout << "Query 정련 작업 완료." << endl;
	return 0;
}

bool haveNum(string & word) {  //string이 알파벳이외의 문자를 가지고 있으면 true 아니면 false


	if (!regex_match(word, pattern)) {
		return true;
	}
	else
		return false;

}
void parse(std::string &word) { //소문자로 바꾸고 불필요한 문자들을 제거 하는 함수

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
cout<<"일치하는 것이 없습니다."<<endl;
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
	
	int docIDtoCompare = 1; //파일에 있는 문서 ID와 비교를 위한 변수
	double sum = 0; //weight계산을 위한 분모를 계산하기 위한 변수
	ifstream Indexed;
	ofstream fout;

	fout.open("sum.dat");
	Indexed.open("indexing.dat");
	if (!Indexed.is_open()) {
		cout << "색인파일을 열지 못했습니다." << endl;
		return 1;
	}
	else {
		cout << "색인파일 오픈 성공" << endl;


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

	cout << "weight 계산을 위한 분모(sum)값 생성 완료" << endl;
	
}


void invert_indexing() {
	
	int wordID = 1; //색인어 ID부여를 위한 변수
	
	double totalRate = 0;
	double weight; //최종 docWeight를 저장할 변수
	ifstream fin;

	ofstream foutI;
	ofstream foutT;
	foutI.open("Inverted_index.dat");
	foutT.open("term.dat");
	fin.open("indexing.dat");

	int starting = 0; //역색인정보 시작위치를 저장할 변수
	
	cout << "역색인 파일 작성중..." << endl;
	
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

			foutI << setw(7) << wordID; //setw(n) 이때 wordID는 7byte를 차지하고 남는 부분을 공백으로 채운다
			foutI << setw(7) << indexInfo[*its][i];  //문서 ID 
			foutI << setw(3) << indexInfo[*its][i+1]; //TF
			foutI << fixed << showpoint << setw(9) << setprecision(4) << weight; //setprecision : 소수점 자리수맞추는 함수 
			//사용시에는 fix -> showpoint -> setw -> setprecision 순으로 사용!!


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

	cout << "길었던 2차텀을 마무리하였다. 고생했다." << endl;

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
				cout << "ParsedNYT/" + NumToString(year) + "/p" + NumToString(year) + strM + strD + "_NYT.txt Indexing시작" << endl;


				if (!fin.is_open()) {
					cout << "파일을 읽지 못했습니다." << endl;
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
								indexInfo[*it].push_back(totalDoc); //docID 넣기
								indexInfo[*it].push_back(oneDocument.count(*it)); //TF 넣기


								CollectionFreq[*it] = CollectionFreq[*it] + oneDocument.count(*it); //CF는 TF들의 합
								DocFreq[*it]++; //DF 증가

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
					cout << "NYT/" + NumToString(year) + "/" + NumToString(year) + strM + strD + "_NYT 완료" << endl;


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
				cout << "ParsedAPW/" + NumToString(year) + "/p" + NumToString(year) + strM + strD + "_APW_ENG.txt Indexing시작" << endl;


				if (!fin.is_open()) {
					cout << "파일을 읽지 못했습니다." << endl;
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


								CollectionFreq[*it] = CollectionFreq[*it] + oneDocument.count(*it); //CF는 TF들의 합
								DocFreq[*it]++; //DF 증가 

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
					cout << "NYT/" + NumToString(year) + "/" + NumToString(year) + strM + strD + "_NYT 완료" << endl;


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
	cout << "indexing완료" << endl;
	return 1;


}
int makestopword() {


	ifstream fin;

	fin.open("stopword.txt");

	if (!fin.is_open()) {
		cout << "파일을 읽지 못했습니다." << endl;
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
				cout << "NYT/" + NumToString(year) + "/" + NumToString(year) + strM + strD + "_NYT 시작" << endl;
				fout.open("ParsedNYT/" + NumToString(year) + "/p" + NumToString(year) + strM + strD + "_NYT.txt");
				int checkH = 0;
				int checkT = 0;
				int checkEnd = 0;
				if (!fin.is_open()) {
					cout << "파일을 읽지 못했습니다." << endl;
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
					cout << "NYT/" + NumToString(year) + "/" + NumToString(year) + strM + strD + "_NYT 완료" << endl;

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
	cout << "NYT stemming 완료" << endl;
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
				cout << "APW/" + NumToString(year) + "/" + NumToString(year) + strM + strD + "_APW_ENG 시작" << endl;
				fout.open("ParsedAPW/" + NumToString(year) + "/p" + NumToString(year) + strM + strD + "_APW_ENG.txt");
				int checkH = 0;
				int checkT = 0;
				int checkEnd = 0;
				if (!fin.is_open()) {
					cout << "파일을 읽지 못했습니다." << endl;
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
				cout << "APW/" + NumToString(year) + "/" + NumToString(year) + strM + strD + "_APW_ENG 완료" << endl;

				if (fout.is_open()) {
					fout.close();
				}
				if (fin.is_open()) {
					fin.close();
				}

			}

		}
	}


	cout << "APW stemming 완료" << endl;
	return 0;
}
