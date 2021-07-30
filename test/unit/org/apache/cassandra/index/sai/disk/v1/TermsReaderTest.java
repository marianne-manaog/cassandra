/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.cassandra.index.sai.disk.v1;

import java.io.IOException;
import java.util.List;

import org.junit.Test;

//<<<<<<< HEAD
import com.carrotsearch.hppc.LongArrayList;
//=======
import org.apache.cassandra.db.marshal.UTF8Type;
import org.apache.cassandra.index.sai.IndexContext;
//>>>>>>> a1c417a8f0 (STAR-158: Add on-disk version support to SAI)
import org.apache.cassandra.index.sai.QueryContext;
import org.apache.cassandra.index.sai.SAITester;
import org.apache.cassandra.index.sai.disk.MemtableTermsIterator;
import org.apache.cassandra.index.sai.disk.PostingList;
import org.apache.cassandra.index.sai.disk.PrimaryKeyMap;
import org.apache.cassandra.index.sai.disk.TermsIterator;
import org.apache.cassandra.index.sai.disk.format.IndexComponent;
import org.apache.cassandra.index.sai.disk.format.IndexDescriptor;
import org.apache.cassandra.index.sai.disk.v1.trie.TermsReader;
import org.apache.cassandra.index.sai.disk.v1.trie.InvertedIndexWriter;
import org.apache.cassandra.index.sai.metrics.QueryEventListener;
import org.apache.cassandra.index.sai.utils.NdiRandomizedTest;
import org.apache.cassandra.index.sai.utils.SAICodecUtils;
import org.apache.cassandra.io.util.FileHandle;
import org.apache.cassandra.utils.Pair;
import org.apache.cassandra.utils.bytecomparable.ByteComparable;
import org.apache.cassandra.utils.bytecomparable.ByteSourceInverse;

import static org.apache.cassandra.index.sai.disk.InvertedIndexBuilder.buildStringTermsEnum;
import static org.apache.cassandra.index.sai.metrics.QueryEventListeners.NO_OP_TRIE_LISTENER;

public class TermsReaderTest extends NdiRandomizedTest
{
    @Test
    public void testTermQueriesAgainstShortPostingLists() throws IOException
    {
        testTermQueries(randomIntBetween(5, 10), randomIntBetween(5, 10));
    }

    @Test
    public void testTermQueriesAgainstLongPostingLists() throws  IOException
    {
        testTermQueries(randomIntBetween(512, 1024), randomIntBetween(1024, 2048));
    }

    @Test
    public void testTermsIteration() throws IOException
    {
        doTestTermsIteration();
    }

    private void doTestTermsIteration() throws IOException
    {
        final int terms = 70, postings = 2;
//<<<<<<< HEAD
//        final IndexComponents indexComponents = newIndexComponents();
//        final List<Pair<ByteComparable, LongArrayList>> termsEnum = buildTermsEnum(terms, postings);
//=======
        final IndexDescriptor indexDescriptor = newIndexDescriptor();
        final String index = newIndex();
        final IndexContext indexContext = SAITester.createIndexContext(index, UTF8Type.instance);
        final List<Pair<ByteComparable, LongArrayList>> termsEnum = buildTermsEnum(terms, postings);
//>>>>>>> a1c417a8f0 (STAR-158: Add on-disk version support to SAI)

        SegmentMetadata.ComponentMetadataMap indexMetas;
        try (InvertedIndexWriter writer = new InvertedIndexWriter(indexDescriptor, index, false))
        {
            indexMetas = writer.writeAll(new MemtableTermsIterator(null, null, termsEnum.iterator()));
        }

        FileHandle termsData = indexDescriptor.createFileHandle(IndexComponent.create(IndexComponent.Type.TERMS_DATA, index));
        FileHandle postingLists = indexDescriptor.createFileHandle(IndexComponent.create(IndexComponent.Type.POSTING_LISTS, index));

        long termsFooterPointer = Long.parseLong(indexMetas.get(IndexComponent.Type.TERMS_DATA).attributes.get(SAICodecUtils.FOOTER_POINTER));

//<<<<<<< HEAD
//        try (TermsReader reader = new TermsReader(indexComponents, null, termsData, postingLists,
//                                                  indexMetas.get(indexComponents.termsData.ndiType).root, termsFooterPointer))
//=======
        try (TermsReader reader = new TermsReader(indexContext,
                                                  null,
                                                  termsData,
                                                  postingLists,
                                                  indexMetas.get(IndexComponent.Type.TERMS_DATA).root,
                                                  termsFooterPointer))
//>>>>>>> a1c417a8f0 (STAR-158: Add on-disk version support to SAI)
        {
            try (TermsIterator actualTermsEnum = reader.allTerms(0, (QueryEventListener.TrieIndexEventListener)NO_OP_TRIE_LISTENER))
            {
                int i = 0;
                for (ByteComparable term = actualTermsEnum.next(); term != null; term = actualTermsEnum.next())
                {
                    final ByteComparable expected = termsEnum.get(i++).left;
                    assertEquals(0, ByteComparable.compare(expected, term, ByteComparable.Version.OSS41));
                }
            }
        }
    }

    private void testTermQueries(int numTerms, int numPostings) throws IOException
    {
        final IndexDescriptor indexDescriptor = newIndexDescriptor();
        final String index = newIndex();
        final IndexContext indexContext = SAITester.createIndexContext(index, UTF8Type.instance);
        final List<Pair<ByteComparable, LongArrayList>> termsEnum = buildTermsEnum(numTerms, numPostings);

        SegmentMetadata.ComponentMetadataMap indexMetas;
        try (InvertedIndexWriter writer = new InvertedIndexWriter(indexDescriptor, index, false))
        {
            indexMetas = writer.writeAll(new MemtableTermsIterator(null, null, termsEnum.iterator()));
        }

        FileHandle termsData = indexDescriptor.createFileHandle(IndexComponent.create(IndexComponent.Type.TERMS_DATA, index));
        FileHandle postingLists = indexDescriptor.createFileHandle(IndexComponent.create(IndexComponent.Type.POSTING_LISTS, index));

        long termsFooterPointer = Long.parseLong(indexMetas.get(IndexComponent.Type.TERMS_DATA).attributes.get(SAICodecUtils.FOOTER_POINTER));

        try (TermsReader reader = new TermsReader(indexContext,
                                                  PrimaryKeyMap.IDENTITY,
                                                  termsData,
                                                  postingLists,
                                                  indexMetas.get(IndexComponent.Type.TERMS_DATA).root,
                                                  termsFooterPointer))
        {
            for (Pair<ByteComparable, LongArrayList> pair : termsEnum)
            {
                final byte[] bytes = ByteSourceInverse.readBytes(pair.left.asComparableBytes(ByteComparable.Version.OSS41));
                try (PostingList actualPostingList = reader.exactMatch(ByteComparable.fixedLength(bytes), (QueryEventListener.TrieIndexEventListener)NO_OP_TRIE_LISTENER, new QueryContext()))
                {
                    final LongArrayList expectedPostingList = pair.right;

                    assertNotNull(actualPostingList);
                    assertEquals(expectedPostingList.size(), actualPostingList.size());

                    for (int i = 0; i < expectedPostingList.size(); ++i)
                    {
                        final long expectedRowID = expectedPostingList.get(i);
                        long result = actualPostingList.nextPosting();
                        assertEquals(expectedRowID, result);
                    }

                    long lastResult = actualPostingList.nextPosting();
                    assertEquals(PostingList.END_OF_STREAM, lastResult);
                }

                // test skipping
                try (PostingList actualPostingList = reader.exactMatch(ByteComparable.fixedLength(bytes), (QueryEventListener.TrieIndexEventListener)NO_OP_TRIE_LISTENER, new QueryContext()))
                {
                    final LongArrayList expectedPostingList = pair.right;
                    // test skipping to the last block
                    final int idxToSkip = numPostings - 2;
                    // tokens are equal to their corresponding row IDs
                    final long tokenToSkip = expectedPostingList.get(idxToSkip);

                    long advanceResult = actualPostingList.advance(PrimaryKeyMap.IDENTITY.primaryKeyFromRowId(tokenToSkip));
                    assertEquals(tokenToSkip, advanceResult);

                    for (int i = idxToSkip + 1; i < expectedPostingList.size(); ++i)
                    {
                        final long expectedRowID = expectedPostingList.get(i);
                        long result = actualPostingList.nextPosting();
                        assertEquals(expectedRowID, result);
                    }

                    long lastResult = actualPostingList.nextPosting();
                    assertEquals(PostingList.END_OF_STREAM, lastResult);
                }
            }
        }
    }

    private List<Pair<ByteComparable, LongArrayList>> buildTermsEnum(int terms, int postings)
    {
        return buildStringTermsEnum(terms, postings, () -> randomSimpleString(4, 10), () -> nextInt(0, Integer.MAX_VALUE));
    }
}
