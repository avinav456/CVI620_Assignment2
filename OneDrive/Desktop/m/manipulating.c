// Disable secure warnings for certain functions
#define _CRT_SECURE_NO_WARNINGS

// Define a constant buffer size
#define BUFFER_SIZE 80

// Include the header file "manipulating.h"
#include "manipulating.h"

// Function to manipulate strings (Version 1)
void manipulating(void)
{
    printf("Manipulation V1\n\n");

    // Print a message indicating the start of the concatenation demo
    printf("*** Start of Concatenating Strings Demo ***\n\n");

    // Declare character arrays to hold input strings
    char string1[BUFFER_SIZE];
    char string2[BUFFER_SIZE];

    // Loop until the user types 'q' to quit
    do
    {
        // Prompt the user to input the first string
        printf("Type the 1st string (q - to quit): \n");

        // Read input from the user and remove the newline character
        fgets(string1, BUFFER_SIZE, stdin);
        string1[strlen(string1) - 1] = '\0';

        // Check if the input string is not 'q'
        if (strcmp(string1, "q") != 0)
        {
            // Prompt the user to input the second string
            printf("Type the 2nd string:\n");
            fgets(string2, BUFFER_SIZE, stdin);
            string2[strlen(string2) - 1] = '\0';

            // Concatenate the second string to the first string
            strcat(string1, string2);

            // Print the concatenated string
            printf("Concatenated string is '%s'\n", string1);
            printf("\n");
        }
    } while (strcmp(string1, "q") != 0);

    // Print a message indicating the end of the concatenation demo
    printf("*** End of Concatenating Strings Demo ***\n\n");


 // Version 2: String Comparison Demo
    printf("Manipulation V2\n\n");

    printf("*** Start of Comparing Strings Demo ***\n\n");
    
    // Declare character arrays to hold input strings for comparison
    char compare1[BUFFER_SIZE];
    char compare2[BUFFER_SIZE];
    
    // Declare a variable to hold the comparison result
    int result;

    // Loop until the user types 'q' to quit
    do {
        // Prompt the user to input the first string for comparison
        printf("Type the 1st string to compare (q - to quit): \n");

        // Read input from the user and remove the newline character
        fgets(compare1, BUFFER_SIZE, stdin);
        compare1[strlen(compare1) - 1] = '\0';

        // Check if the input string is not 'q'
        if (strcmp(compare1, "q") != 0) {
            // Prompt the user to input the second string for comparison
            printf("Type the 2nd string to compare: \n");
            fgets(compare2, BUFFER_SIZE, stdin);
            compare2[strlen(compare2) - 1] = '\0';

            // Compare the two strings and store the result
            result = strcmp(compare1, compare2);

            // Check the comparison result and print appropriate messages
            if (result < 0)
                printf("'%s' string is less than '%s'\n", compare1, compare2);
            else if (result == 0)
                printf("'%s' string is equal to '%s'\n", compare1, compare2);
            else
                printf("'%s' string is greater than '%s'\n", compare1, compare2);
        }
        printf("\n");
    } while (strcmp(compare1, "q") != 0);

    // Print a message indicating the end of the string comparison demo
    printf("*** End of Comparing Strings Demo ***\n\n");

       // Version 3 Searching Strings Demo

    // Presentation message
    printf("Manipulation V3\n\n"); // Print a message indicating the version 3 of string manipulation

    printf("*** Start of Searching Strings Demo ***\n\n"); // Print a message indicating the beginning of the string searching demo

    // Variables declaration
    char haystack[BUFFER_SIZE]; // Declaration of a variable to store the string where we'll perform the search
    char needle[BUFFER_SIZE]; // Declaration of a variable to store the substring we want to search for
    char* occurrence = NULL; // Declaration of a char pointer that will be used to store the occurrence of the substring

    // Main loop
    do {

        printf("Type the string (q - to quit) : \n");// Prompt the user to type the string for the search

        fgets(haystack, BUFFER_SIZE, stdin); // Read user input and store it in the variable haystack

        haystack[strlen(haystack) - 1] = '\0'; // Replace the newline character (\n) with the character '0'

        if (strcmp(haystack, "q") != 0) { // Check if the user entered 'q' to quit

            printf("Type the substring:\n"); // Prompt the user to type the substring we want to search for

            fgets(needle, BUFFER_SIZE, stdin); // Read user input and store it in the variable needle

            needle[strlen(needle) - 1] = '\0'; // Remove the newline character (\n) from the substring

            occurrence = strstr(haystack, needle); // Search for the occurrence of the substring in the haystack

            if (occurrence) { // If the occurrence is found
                printf("\'%s\' found at %d position \n", needle, (int)(occurrence - haystack)); // Print the found substring and its position in the haystack
            }
            else {
                printf("Not found\n"); // Otherwise, print that the substring was not found
            }
        }
        printf("\n");
    } while (strcmp(haystack, "q") != 0); // Continue the loop until the user types 'q'

    printf("*** End of Searching Strings Demo ***\n\n"); // Print a message indicating the end of the string searching demolea

}